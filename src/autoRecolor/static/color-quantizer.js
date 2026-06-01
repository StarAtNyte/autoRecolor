class ColorQuantizer {
    static SUPPORTED_DITHER_COLOR_COUNTS = [2];
    static SUPPORTED_MIX_PLY_VALUES = [2, 3, 4];
    static TOP_COLOR_SAMPLE_SIZE = 100;
    static MAX_KMEANS_ITERATIONS = 24;
    static PROGRESS_YIELD_INTERVAL = 24;
    static MAX_MIX_ANCHORS = 9;
    static MAX_DITHER_OKLAB_DISTANCE = 0.035;
    static MIN_DITHER_DISTANCE_IMPROVEMENT = 0.002;
    static ACHROMATIC_EPSILON = 1e-7;
    static TWO_COLOR_STEPS = [
        { ratioA: 1, thresholdCountA: 12, label: '100/0' },
        { ratioA: 0.75, thresholdCountA: 9, label: '75/25' },
        { ratioA: 2 / 3, thresholdCountA: 8, label: '67/33' },
        { ratioA: 0.5, thresholdCountA: 6, label: '50/50' },
        { ratioA: 1 / 3, thresholdCountA: 4, label: '33/67' },
        { ratioA: 0.25, thresholdCountA: 3, label: '25/75' },
        { ratioA: 0, thresholdCountA: 0, label: '0/100' }
    ];

    static async quantizeImage(imageData, targetColors, onProgress = async () => {}) {
        if (!imageData || !imageData.data || !imageData.width || !imageData.height) {
            throw new Error('ImageData is required for quantization.');
        }

        const safeTargetColors = Math.max(1, Math.min(100, Math.round(targetColors)));
        const width = imageData.width;
        const height = imageData.height;
        const totalPixels = width * height;

        await onProgress(8, 'Sampling candidate colors...');
        const { topColors, sampleCount, uniqueSampleColors } = this._findTopColors(imageData, safeTargetColors);

        await onProgress(
            20,
            `Running k-means for ${topColors.length} color${topColors.length === 1 ? '' : 's'} on ${totalPixels.toLocaleString()} pixels...`
        );
        const { palette, iterations } = await this._runKMeans(imageData, topColors, onProgress);

        await onProgress(82, 'Mapping pixels to the quantized palette...');
        const { output, colorUsage, pixelIndices } = await this._applyPalette(imageData, palette, onProgress);

        await onProgress(100, 'Done.');

        return {
            width,
            height,
            imageData: output,
            pixelIndices,
            palette,
            stats: {
                requestedColors: safeTargetColors,
                centroidCount: palette.length,
                actualColors: colorUsage.length,
                totalPixels,
                sampleCount,
                uniqueSampleColors,
                trainingPixelCount: totalPixels,
                trainingStep: 1,
                iterations,
                colorUsage
            }
        };
    }

    static async detectRepresentativeColors(imageData, targetColors = 7, onProgress = async () => {}) {
        if (!imageData || !imageData.data || !imageData.width || !imageData.height) {
            throw new Error('ImageData is required for representative color detection.');
        }

        const safeTargetColors = Math.max(1, Math.min(100, Math.round(targetColors)));
        const width = imageData.width;
        const height = imageData.height;
        const totalPixels = width * height;

        await onProgress(8, 'Sampling candidate colors...');
        const { topColors, sampleCount, uniqueSampleColors } = this._findTopColors(imageData, safeTargetColors);

        await onProgress(
            20,
            `Running k-means for ${topColors.length} representative color${topColors.length === 1 ? '' : 's'}...`
        );
        const { palette, iterations } = await this._runKMeans(imageData, topColors, onProgress);

        await onProgress(100, 'Representative colors ready.');

        return {
            width,
            height,
            palette,
            stats: {
                requestedColors: safeTargetColors,
                centroidCount: palette.length,
                totalPixels,
                sampleCount,
                uniqueSampleColors,
                trainingPixelCount: totalPixels,
                trainingStep: 1,
                iterations
            }
        };
    }

    static isDitherColorCountSupported(ditherColorCount) {
        return this.SUPPORTED_DITHER_COLOR_COUNTS.includes(Math.round(ditherColorCount));
    }

    static createQuantizedDither(quantization, ditherColorCount = 2, mixPly = true) {
        if (!quantization || !Array.isArray(quantization.palette) || !quantization.pixelIndices) {
            throw new Error('Quantization data with palette indices is required for dithering.');
        }

        if (!this.isDitherColorCountSupported(ditherColorCount)) {
            return {
                implemented: false,
                ditherColorCount,
                reason: 'K-means currently supports 2-color stochastic dithering only.'
            };
        }

        const plan = this._buildTwoColorDitherPlan(quantization);
        const imageData = mixPly
            ? this._renderTwoColorDither(quantization, plan)
            : this._renderTwoColorBlend(quantization, plan);

        return {
            implemented: true,
            ditherColorCount,
            mixPly,
            imageData,
            plan
        };
    }

    static async renderColorAnything(sourceImageData, quantization, options = {}, onProgress = async () => {}) {
        if (!sourceImageData || !sourceImageData.data || !sourceImageData.width || !sourceImageData.height) {
            throw new Error('Source ImageData is required for K-means palette rendering.');
        }

        if (!quantization || !Array.isArray(quantization.palette) || quantization.palette.length === 0) {
            throw new Error('Quantization data with a centroid palette is required for rendering.');
        }

        if (sourceImageData.width !== quantization.width || sourceImageData.height !== quantization.height) {
            throw new Error('Source image and quantization dimensions do not match.');
        }

        const resolved = this._normalizeRenderOptions(options);
        const paletteModel = await this._getPaletteModel(sourceImageData, quantization, resolved, onProgress);
        const totalPixels = sourceImageData.width * sourceImageData.height;

        await onProgress(68, 'Applying merge rules...');
        const remap = this._buildIndexRemap(
            paletteModel.paletteLab,
            paletteModel.rawCounts,
            Math.max(totalPixels, 1),
            resolved
        );

        const finalCounts = new Uint32Array(paletteModel.palette.length);
        for (let index = 0; index < paletteModel.pixelIndices.length; index += 1) {
            finalCounts[remap[paletteModel.pixelIndices[index]]] += 1;
        }

        await onProgress(78, resolved.overlayEnabled ? 'Rendering mapped palette with lightness overlay...' : 'Rendering mapped palette...');
        const imageData = await this._renderMappedOutput(
            sourceImageData,
            paletteModel,
            remap,
            resolved,
            onProgress
        );

        const usageEntries = this._buildUsageEntries(
            paletteModel.palette,
            finalCounts,
            Math.max(totalPixels, 1),
            null,
            paletteModel.mixInfo,
            false
        );

        const solidEntryCount = usageEntries.filter(entry => entry.kind === 'representative').length;
        const blendedEntryCount = usageEntries.filter(entry => entry.kind === 'interpolated').length;
        const ditheredEntryCount = 0;

        await onProgress(100, 'Done.');

        return {
            width: sourceImageData.width,
            height: sourceImageData.height,
            imageData,
            palette: paletteModel.palette,
            counts: finalCounts,
            usageEntries,
            mapping: {
                pixelIndices: paletteModel.pixelIndices,
                remap,
                mixInfo: paletteModel.mixInfo,
                pairCount: paletteModel.pairCount,
                generatedInterpolationCount: paletteModel.generatedInterpolationCount
            },
            stats: {
                totalPixels,
                usedPaletteEntries: usageEntries.length,
                representativeEntryCount: solidEntryCount,
                interpolatedEntryCount: blendedEntryCount + ditheredEntryCount,
                totalPaletteEntries: paletteModel.palette.length,
                baseAnchorCount: quantization.palette.length,
                pairCount: paletteModel.pairCount,
                generatedInterpolationCount: paletteModel.generatedInterpolationCount
            }
        };
    }

    static recolorWithExistingMapping(sourceImageData, quantization, previousRenderResult, options = {}) {
        if (!sourceImageData || !sourceImageData.data || !quantization || !Array.isArray(quantization.palette)) {
            throw new Error('Source image data and quantization palette are required for recoloring.');
        }

        const mapping = previousRenderResult?.mapping;
        if (!mapping || !mapping.pixelIndices || !mapping.remap || !mapping.mixInfo) {
            throw new Error('Previous render mapping is required for recoloring.');
        }

        const resolved = this._normalizeRenderOptions(options);
        const palette = this._buildPaletteFromExistingTopology(
            quantization.palette,
            mapping.mixInfo,
            mapping.remap.length
        );
        const paletteLab = palette.map(color => this._rgbToOklab(color[0], color[1], color[2]));
        const imageData = this._renderMappedOutputSync(
            sourceImageData,
            {
                palette,
                paletteLab,
                pixelIndices: mapping.pixelIndices
            },
            mapping.remap,
            resolved
        );

        const counts = new Uint32Array(mapping.remap.length);
        for (let index = 0; index < mapping.pixelIndices.length; index += 1) {
            counts[mapping.remap[mapping.pixelIndices[index]]] += 1;
        }

        const usageEntries = this._buildUsageEntries(
            palette,
            counts,
            Math.max(sourceImageData.width * sourceImageData.height, 1),
            null,
            mapping.mixInfo,
            false
        );
        const representativeEntryCount = usageEntries.filter(entry => entry.kind === 'representative').length;
        const interpolatedEntryCount = usageEntries.filter(entry => entry.kind === 'interpolated').length;

        return {
            width: sourceImageData.width,
            height: sourceImageData.height,
            imageData,
            palette,
            counts,
            usageEntries,
            mapping,
            stats: {
                totalPixels: sourceImageData.width * sourceImageData.height,
                usedPaletteEntries: usageEntries.length,
                representativeEntryCount,
                interpolatedEntryCount,
                totalPaletteEntries: palette.length,
                baseAnchorCount: quantization.palette.length,
                pairCount: mapping.pairCount || 0,
                generatedInterpolationCount: mapping.generatedInterpolationCount || mapping.mixInfo.size
            }
        };
    }

    static replacePaletteEntryWithExistingMapping(sourceImageData, previousRenderResult, paletteIndex, nextColor, options = {}) {
        if (!sourceImageData || !sourceImageData.data) {
            throw new Error('Source image data is required for direct palette replacement.');
        }

        if (!previousRenderResult || !Array.isArray(previousRenderResult.palette)) {
            throw new Error('Previous render result is required for direct palette replacement.');
        }

        const mapping = previousRenderResult.mapping;
        if (!mapping || !mapping.pixelIndices || !mapping.remap) {
            throw new Error('Previous render mapping is required for direct palette replacement.');
        }

        if (!Number.isInteger(paletteIndex) || paletteIndex < 0 || paletteIndex >= previousRenderResult.palette.length) {
            throw new Error('Palette index is out of range for direct palette replacement.');
        }

        const resolved = this._normalizeRenderOptions(options);
        const palette = previousRenderResult.palette.map(color => [...color]);
        palette[paletteIndex] = [
            Math.max(0, Math.min(255, Math.round(nextColor[0] ?? 0))),
            Math.max(0, Math.min(255, Math.round(nextColor[1] ?? 0))),
            Math.max(0, Math.min(255, Math.round(nextColor[2] ?? 0)))
        ];
        const paletteLab = palette.map(color => this._rgbToOklab(color[0], color[1], color[2]));
        const imageData = this._renderMappedOutputSync(
            sourceImageData,
            {
                palette,
                paletteLab,
                pixelIndices: mapping.pixelIndices
            },
            mapping.remap,
            resolved
        );

        const counts = previousRenderResult.counts
            ? new Uint32Array(previousRenderResult.counts)
            : this._countMappedPixels(mapping.pixelIndices, mapping.remap, palette.length);
        const usageEntries = this._buildUsageEntries(
            palette,
            counts,
            Math.max(sourceImageData.width * sourceImageData.height, 1),
            null,
            mapping.mixInfo || new Map(),
            false
        );
        const representativeEntryCount = usageEntries.filter(entry => entry.kind === 'representative').length;
        const interpolatedEntryCount = usageEntries.filter(entry => entry.kind === 'interpolated').length;

        return {
            width: sourceImageData.width,
            height: sourceImageData.height,
            imageData,
            palette,
            counts,
            usageEntries,
            mapping,
            stats: {
                totalPixels: sourceImageData.width * sourceImageData.height,
                usedPaletteEntries: usageEntries.length,
                representativeEntryCount,
                interpolatedEntryCount,
                totalPaletteEntries: palette.length,
                baseAnchorCount: previousRenderResult.stats?.baseAnchorCount || 0,
                pairCount: previousRenderResult.stats?.pairCount || 0,
                generatedInterpolationCount: previousRenderResult.stats?.generatedInterpolationCount || 0
            }
        };
    }

    static _normalizeRenderOptions(options = {}) {
        const threshold = Number.isFinite(options.mergeThreshold)
            ? Math.min(1, Math.max(0, options.mergeThreshold))
            : 0.005;
        const interpolationEnabled = options.interpolationEnabled !== false;
        const interpolationSteps = Number.isFinite(options.interpolationSteps)
            ? Math.max(1, Math.min(20, Math.round(options.interpolationSteps)))
            : 6;

        return {
            overlayEnabled: options.overlayEnabled === true,
            mergeEnabled: options.mergeEnabled !== false,
            mergeThreshold: threshold,
            interpolationEnabled,
            interpolationSteps
        };
    }

    static async _getPaletteModel(sourceImageData, quantization, options, onProgress) {
        const cacheKey = this._getPaletteModelCacheKey(options);
        quantization.renderCache = quantization.renderCache || new Map();
        if (quantization.renderCache.has(cacheKey)) {
            await onProgress(18, 'Reusing cached palette model...');
            return quantization.renderCache.get(cacheKey);
        }

        await onProgress(8, 'Building palette model...');
        const { palette, mixInfo, generatedInterpolationCount, pairCount } = this._buildExpandedPalette(
            quantization.palette,
            options.interpolationSteps,
            options.interpolationEnabled
        );
        const paletteLab = palette.map(color => this._rgbToOklab(color[0], color[1], color[2]));
        const lut = await this._buildIndexLut(paletteLab, onProgress);
        const { pixelIndices, rawCounts } = await this._mapPixelsToPalette(sourceImageData, lut, palette.length, onProgress);

        const paletteModel = {
            palette,
            paletteLab,
            mixInfo,
            generatedInterpolationCount,
            pairCount,
            lut,
            pixelIndices,
            rawCounts
        };

        quantization.renderCache.set(cacheKey, paletteModel);
        return paletteModel;
    }

    static _getPaletteModelCacheKey(options) {
        return [
            options.interpolationEnabled ? 1 : 0,
            options.interpolationSteps
        ].join('|');
    }

    static buildInterpolatedPalette(basePalette, interpolationSteps = 6, interpolationEnabled = true) {
        const safeSteps = Math.max(1, Math.min(20, Math.round(interpolationSteps)));
        const palette = basePalette.map(color => [...color]);
        const anchorLabs = basePalette.map(color => this._rgbToOklab(color[0], color[1], color[2]));
        const mixInfo = new Map();
        let generatedInterpolationCount = 0;
        let pairCount = 0;
        const seen = new Set(palette.map(color => this._toHex(color)));

        if (!interpolationEnabled || basePalette.length < 2) {
            return { palette, mixInfo, generatedInterpolationCount, pairCount };
        }

        for (let left = 0; left < basePalette.length; left += 1) {
            for (let right = left + 1; right < basePalette.length; right += 1) {
                pairCount += 1;
                const leftLab = anchorLabs[left];
                const rightLab = anchorLabs[right];

                for (let step = 1; step <= safeSteps; step += 1) {
                    const t = step / (safeSteps + 1);
                    const color = this._oklabToRgb({
                        l: leftLab.l + ((rightLab.l - leftLab.l) * t),
                        a: leftLab.a + ((rightLab.a - leftLab.a) * t),
                        b: leftLab.b + ((rightLab.b - leftLab.b) * t)
                    });
                    const key = this._toHex(color);
                    if (seen.has(key)) {
                        continue;
                    }

                    seen.add(key);
                    const paletteIndex = palette.length;
                    palette.push(color);
                    mixInfo.set(paletteIndex, {
                        leftIndex: left,
                        rightIndex: right,
                        step,
                        totalSteps: safeSteps
                    });
                    generatedInterpolationCount += 1;
                }
            }
        }

        return { palette, mixInfo, generatedInterpolationCount, pairCount };
    }

    static _buildExpandedPalette(basePalette, interpolationSteps, interpolationEnabled = true) {
        return this.buildInterpolatedPalette(basePalette, interpolationSteps, interpolationEnabled);
    }

    static async _buildIndexLut(paletteLab, onProgress) {
        const size = 32;
        const lut = new Uint16Array(size * size * size);

        for (let redIndex = 0; redIndex < size; redIndex += 1) {
            for (let greenIndex = 0; greenIndex < size; greenIndex += 1) {
                for (let blueIndex = 0; blueIndex < size; blueIndex += 1) {
                    const sample = this._rgbToOklab(redIndex * 8, greenIndex * 8, blueIndex * 8);
                    let bestIndex = 0;
                    let bestDistance = Infinity;

                    for (let paletteIndex = 0; paletteIndex < paletteLab.length; paletteIndex += 1) {
                        const distance = this._oklabDistanceSquared(sample, paletteLab[paletteIndex]);
                        if (distance < bestDistance) {
                            bestDistance = distance;
                            bestIndex = paletteIndex;
                        }
                    }

                    lut[(redIndex * size * size) + (greenIndex * size) + blueIndex] = bestIndex;
                }
            }

            if (redIndex % 4 === 0 || redIndex === size - 1) {
                const progress = 12 + Math.round(((redIndex + 1) / size) * 24);
                await onProgress(progress, `Building palette lookup (${redIndex + 1}/${size})...`);
            }
        }

        return lut;
    }

    static async _mapPixelsToPalette(imageData, lut, paletteLength, onProgress) {
        const { width, height, data } = imageData;
        const pixelIndices = new Uint16Array(width * height);
        const rawCounts = new Uint32Array(paletteLength);

        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                const pixelIndex = (y * width) + x;
                const offset = pixelIndex * 4;
                const redIndex = Math.min(31, data[offset] >> 3);
                const greenIndex = Math.min(31, data[offset + 1] >> 3);
                const blueIndex = Math.min(31, data[offset + 2] >> 3);
                const paletteIndex = lut[(redIndex * 1024) + (greenIndex * 32) + blueIndex];

                pixelIndices[pixelIndex] = paletteIndex;
                rawCounts[paletteIndex] += 1;
            }

            if (y % this.PROGRESS_YIELD_INTERVAL === 0 || y === height - 1) {
                const progress = 38 + Math.round(((y + 1) / height) * 24);
                await onProgress(progress, `Assigning pixels to the expanded palette (${y + 1}/${height})...`);
            }
        }

        return { pixelIndices, rawCounts };
    }

    static _buildIndexRemap(paletteLab, counts, totalPixels, options) {
        const remap = new Uint16Array(paletteLab.length);
        for (let index = 0; index < paletteLab.length; index += 1) {
            remap[index] = index;
        }

        if (!options.mergeEnabled) {
            return remap;
        }

        for (let index = 0; index < paletteLab.length; index += 1) {
            if ((counts[index] / totalPixels) >= options.mergeThreshold) {
                continue;
            }

            let bestIndex = -1;
            let bestDistance = Infinity;
            for (let candidate = 0; candidate < paletteLab.length; candidate += 1) {
                if ((counts[candidate] / totalPixels) < options.mergeThreshold) {
                    continue;
                }

                const distance = this._oklabDistanceSquared(paletteLab[index], paletteLab[candidate]);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestIndex = candidate;
                }
            }

            if (bestIndex >= 0) {
                remap[index] = bestIndex;
            }
        }

        return remap;
    }

    static async _renderMappedOutput(sourceImageData, paletteModel, remap, options, onProgress) {
        const { width, height, data } = sourceImageData;
        const output = new Uint8ClampedArray(data.length);

        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                const pixelIndex = (y * width) + x;
                const offset = pixelIndex * 4;
                const mappedIndex = remap[paletteModel.pixelIndices[pixelIndex]];
                const colorIndex = mappedIndex;
                const color = paletteModel.palette[mappedIndex] || [0, 0, 0];

                if (options.overlayEnabled) {
                    const originalLab = this._rgbToOklab(data[offset], data[offset + 1], data[offset + 2]);
                    const overlaid = this._oklabToRgb(
                        this._buildRenderedOklab(paletteModel.paletteLab[colorIndex], originalLab, options)
                    );
                    output[offset] = overlaid[0];
                    output[offset + 1] = overlaid[1];
                    output[offset + 2] = overlaid[2];
                } else {
                    output[offset] = color[0];
                    output[offset + 1] = color[1];
                    output[offset + 2] = color[2];
                }

                output[offset + 3] = data[offset + 3];
            }

            if (y % this.PROGRESS_YIELD_INTERVAL === 0 || y === height - 1) {
                const progress = 80 + Math.round(((y + 1) / height) * 18);
                await onProgress(progress, `Painting output pixels (${y + 1}/${height})...`);
            }
        }

        return output;
    }

    static _renderMappedOutputSync(sourceImageData, paletteModel, remap, options) {
        const { width, height, data } = sourceImageData;
        const output = new Uint8ClampedArray(data.length);

        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                const pixelIndex = (y * width) + x;
                const offset = pixelIndex * 4;
                const mappedIndex = remap[paletteModel.pixelIndices[pixelIndex]];
                const color = paletteModel.palette[mappedIndex] || [0, 0, 0];

                if (options.overlayEnabled) {
                    const originalLab = this._rgbToOklab(data[offset], data[offset + 1], data[offset + 2]);
                    const overlaid = this._oklabToRgb(
                        this._buildRenderedOklab(paletteModel.paletteLab[mappedIndex], originalLab, options)
                    );
                    output[offset] = overlaid[0];
                    output[offset + 1] = overlaid[1];
                    output[offset + 2] = overlaid[2];
                } else {
                    output[offset] = color[0];
                    output[offset + 1] = color[1];
                    output[offset + 2] = color[2];
                }

                output[offset + 3] = data[offset + 3];
            }
        }

        return output;
    }

    static _buildRenderedOklab(mappedLab, originalLab, options) {
        const lightness = options.overlayEnabled && originalLab ? originalLab.l : mappedLab.l;
        return {
            l: lightness,
            a: mappedLab.a,
            b: mappedLab.b
        };
    }

    static _buildUsageEntries(palette, counts, totalPixels, ditherInfo, mixInfo, mixPlyEnabled) {
        return palette
            .map((color, index) => ({ color, index, count: counts[index] }))
            .filter(entry => entry.count > 0)
            .sort((left, right) => right.count - left.count || left.index - right.index)
            .map(entry => {
                const mix = mixInfo.get(entry.index) || null;
                const kind = mix ? 'interpolated' : 'representative';
                const mixParts = mix
                    ? [
                        {
                            countLabel: `A`,
                            color: palette[mix.leftIndex],
                            label: this._formatOklchLabel(palette[mix.leftIndex])
                        },
                        {
                            countLabel: `${mix.step}/${mix.totalSteps + 1}`,
                            color: entry.color,
                            label: this._formatOklchLabel(entry.color)
                        },
                        {
                            countLabel: `B`,
                            color: palette[mix.rightIndex],
                            label: this._formatOklchLabel(palette[mix.rightIndex])
                        }
                    ]
                    : [];

                return {
                    index: entry.index,
                    color: entry.color,
                    hex: this._toHex(entry.color),
                    count: entry.count,
                    percent: (entry.count / totalPixels) * 100,
                    kind,
                    mixParts,
                    swatchDataUrl: ''
                };
            });
    }

    static _countMappedPixels(pixelIndices, remap, paletteLength) {
        const counts = new Uint32Array(paletteLength);
        for (let index = 0; index < pixelIndices.length; index += 1) {
            counts[remap[pixelIndices[index]]] += 1;
        }
        return counts;
    }

    static createDitherSwatchDataUrl(parts, palette, totalPly, seed = 0) {
        if (typeof document === 'undefined') {
            return '';
        }

        const width = 56;
        const height = 56;
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d');
        if (!context) {
            return '';
        }

        const imageData = context.createImageData(width, height);
        const output = imageData.data;
        for (let index = 0; index < width * height; index += 1) {
            const x = index % width;
            const y = Math.floor(index / width);
            const colorIndex = this._resolveDitherColorIndex(parts, totalPly, parts[0]?.anchorIdx ?? 0, x, y, seed);
            const color = palette[colorIndex] || [0, 0, 0];
            output[index * 4] = color[0];
            output[index * 4 + 1] = color[1];
            output[index * 4 + 2] = color[2];
            output[index * 4 + 3] = 255;
        }

        context.putImageData(imageData, 0, 0);
        return canvas.toDataURL();
    }

    static _resolveDitherColorIndex(parts, totalPly, fallbackIndex, x, y, label) {
        let cumulative = 0;
        const threshold = this._pixelNoise(x, y, label) * totalPly;

        for (const part of parts) {
            cumulative += part.count;
            if (threshold < cumulative) {
                return part.anchorIdx;
            }
        }

        return parts[parts.length - 1]?.anchorIdx ?? fallbackIndex;
    }

    static _formatOklchLabel(color) {
        const oklch = this._oklabToOklch(this._rgbToOklab(color[0], color[1], color[2]));
        return `L ${(oklch.l * 100).toFixed(1)} | C ${oklch.c.toFixed(3)} | H ${(oklch.h * 180 / Math.PI).toFixed(1)}`;
    }

    static _buildPaletteFromExistingTopology(basePalette, mixInfo, paletteLength) {
        const palette = new Array(paletteLength);
        for (let index = 0; index < basePalette.length && index < palette.length; index += 1) {
            palette[index] = [...basePalette[index]];
        }

        for (const [paletteIndex, mix] of mixInfo.entries()) {
            const leftLab = this._rgbToOklab(basePalette[mix.leftIndex][0], basePalette[mix.leftIndex][1], basePalette[mix.leftIndex][2]);
            const rightLab = this._rgbToOklab(basePalette[mix.rightIndex][0], basePalette[mix.rightIndex][1], basePalette[mix.rightIndex][2]);
            const t = mix.step / (mix.totalSteps + 1);
            palette[paletteIndex] = this._oklabToRgb({
                l: leftLab.l + ((rightLab.l - leftLab.l) * t),
                a: leftLab.a + ((rightLab.a - leftLab.a) * t),
                b: leftLab.b + ((rightLab.b - leftLab.b) * t)
            });
        }

        for (let index = 0; index < palette.length; index += 1) {
            if (!palette[index]) {
                palette[index] = [...basePalette[Math.min(index, basePalette.length - 1)]];
            }
        }

        return palette;
    }

    static _buildTwoColorDitherPlan(quantization) {
        const paletteInfo = this._buildPaletteInfo(quantization);
        const activeIndices = paletteInfo
            .filter(entry => entry.count > 0)
            .map(entry => entry.index);
        const activeIndexSet = new Set(activeIndices);
        const protectedIndices = new Set();
        const pairMixes = this._buildTwoColorPairMixes(activeIndices, paletteInfo);
        const replacementByIndex = new Array(quantization.palette.length).fill(null);
        const replacements = [];

        while (true) {
            let bestReplacement = null;

            for (const targetIndex of activeIndices) {
                if (protectedIndices.has(targetIndex)) {
                    continue;
                }

                const candidate = this._findBestTwoColorReplacement(
                    targetIndex,
                    activeIndexSet,
                    paletteInfo,
                    pairMixes
                );

                if (!candidate) {
                    continue;
                }

                if (candidate.distance > this.MAX_DITHER_OKLAB_DISTANCE) {
                    continue;
                }

                if (candidate.distance + this.MIN_DITHER_DISTANCE_IMPROVEMENT >= candidate.nearestSolidDistance) {
                    continue;
                }

                if (
                    !bestReplacement ||
                    candidate.distance < bestReplacement.distance - Number.EPSILON ||
                    (
                        Math.abs(candidate.distance - bestReplacement.distance) <= Number.EPSILON &&
                        candidate.targetCount < bestReplacement.targetCount
                    ) ||
                    (
                        Math.abs(candidate.distance - bestReplacement.distance) <= Number.EPSILON &&
                        candidate.targetCount === bestReplacement.targetCount &&
                        candidate.targetIndex < bestReplacement.targetIndex
                    )
                ) {
                    bestReplacement = candidate;
                }
            }

            if (!bestReplacement) {
                break;
            }

            activeIndexSet.delete(bestReplacement.targetIndex);
            const removalIndex = activeIndices.indexOf(bestReplacement.targetIndex);
            if (removalIndex >= 0) {
                activeIndices.splice(removalIndex, 1);
            }

            protectedIndices.add(bestReplacement.baseAIndex);
            protectedIndices.add(bestReplacement.baseBIndex);
            replacementByIndex[bestReplacement.targetIndex] = bestReplacement;
            replacements.push(bestReplacement);
        }

        activeIndices.sort((left, right) => (
            paletteInfo[right].count - paletteInfo[left].count ||
            left - right
        ));
        replacements.sort((left, right) => left.targetIndex - right.targetIndex);

        return {
            originalColorCount: paletteInfo.filter(entry => entry.count > 0).length,
            solidColorCount: activeIndices.length,
            replacedColorCount: replacements.length,
            basePaletteIndices: activeIndices,
            replacements,
            replacementByIndex
        };
    }

    static _buildPaletteInfo(quantization) {
        const totalPixels = quantization.width * quantization.height;
        const usageByIndex = new Uint32Array(quantization.palette.length);

        if (quantization.stats && Array.isArray(quantization.stats.colorUsage)) {
            for (const entry of quantization.stats.colorUsage) {
                if (entry && Number.isInteger(entry.index) && entry.index >= 0 && entry.index < usageByIndex.length) {
                    usageByIndex[entry.index] = entry.count;
                }
            }
        } else {
            for (let pixelIndex = 0; pixelIndex < quantization.pixelIndices.length; pixelIndex += 1) {
                usageByIndex[quantization.pixelIndices[pixelIndex]] += 1;
            }
        }

        return quantization.palette.map((rgb, index) => {
            const oklab = this._rgbToOklab(rgb[0], rgb[1], rgb[2]);
            const oklch = this._oklabToOklch(oklab);
            return {
                index,
                rgb,
                hex: this._toHex(rgb),
                count: usageByIndex[index],
                percent: totalPixels > 0 ? (usageByIndex[index] / totalPixels) * 100 : 0,
                oklab,
                oklch
            };
        });
    }

    static _buildTwoColorPairMixes(activeIndices, paletteInfo) {
        const mixes = [];

        for (let firstIndex = 0; firstIndex < activeIndices.length; firstIndex += 1) {
            const baseAIndex = activeIndices[firstIndex];
            for (let secondIndex = firstIndex + 1; secondIndex < activeIndices.length; secondIndex += 1) {
                const baseBIndex = activeIndices[secondIndex];
                const steps = [];

                for (const step of this.TWO_COLOR_STEPS) {
                    if (step.ratioA <= 0 || step.ratioA >= 1) {
                        continue;
                    }

                    const mixedOklch = this._mixOklch(
                        paletteInfo[baseAIndex].oklch,
                        paletteInfo[baseBIndex].oklch,
                        step.ratioA
                    );
                    const mixedOklab = this._oklchToOklab(mixedOklch);
                    steps.push({
                        ratioA: step.ratioA,
                        ratioB: 1 - step.ratioA,
                        thresholdCountA: step.thresholdCountA,
                        label: step.label,
                        mixedOklab,
                        mixedRgb: this._oklabToRgb(mixedOklab)
                    });
                }

                mixes.push({
                    baseAIndex,
                    baseBIndex,
                    steps
                });
            }
        }

        return mixes;
    }

    static _findBestTwoColorReplacement(targetIndex, activeIndexSet, paletteInfo, pairMixes) {
        const target = paletteInfo[targetIndex];
        let nearestSolidDistance = Infinity;

        for (const other of paletteInfo) {
            if (!activeIndexSet.has(other.index) || other.index === targetIndex) {
                continue;
            }

            nearestSolidDistance = Math.min(
                nearestSolidDistance,
                this._oklabDistance(target.oklab, other.oklab)
            );
        }

        if (!Number.isFinite(nearestSolidDistance)) {
            return null;
        }

        let best = null;

        for (const pair of pairMixes) {
            if (!activeIndexSet.has(pair.baseAIndex) || !activeIndexSet.has(pair.baseBIndex)) {
                continue;
            }

            if (pair.baseAIndex === targetIndex || pair.baseBIndex === targetIndex) {
                continue;
            }

            for (const step of pair.steps) {
                const distance = this._oklabDistance(target.oklab, step.mixedOklab);
                if (
                    !best ||
                    distance < best.distance - Number.EPSILON ||
                    (
                        Math.abs(distance - best.distance) <= Number.EPSILON &&
                        step.thresholdCountA > best.thresholdCountA
                    )
                ) {
                    best = {
                        targetIndex,
                        targetHex: target.hex,
                        targetCount: target.count,
                        targetPercent: target.percent,
                        baseAIndex: pair.baseAIndex,
                        baseBIndex: pair.baseBIndex,
                        baseAHex: paletteInfo[pair.baseAIndex].hex,
                        baseBHex: paletteInfo[pair.baseBIndex].hex,
                        ratioA: step.ratioA,
                        ratioB: step.ratioB,
                        thresholdCountA: step.thresholdCountA,
                        ratioLabel: step.label,
                        mixedRgb: step.mixedRgb,
                        mixedHex: this._toHex(step.mixedRgb),
                        distance,
                        nearestSolidDistance
                    };
                }
            }
        }

        return best;
    }

    static _renderTwoColorDither(quantization, plan) {
        const { width, height, pixelIndices, palette, imageData } = quantization;
        const output = new Uint8ClampedArray(width * height * 4);
        const replacementByIndex = plan.replacementByIndex;

        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                const pixelIndex = (y * width) + x;
                const outputOffset = pixelIndex * 4;
                const sourcePaletteIndex = pixelIndices[pixelIndex];
                const replacement = replacementByIndex[sourcePaletteIndex];
                let rgb = palette[sourcePaletteIndex];

                if (replacement) {
                    const noise = this._pixelNoise(x, y, sourcePaletteIndex);
                    rgb = noise < replacement.ratioA
                        ? palette[replacement.baseAIndex]
                        : palette[replacement.baseBIndex];
                }

                output[outputOffset] = rgb[0];
                output[outputOffset + 1] = rgb[1];
                output[outputOffset + 2] = rgb[2];
                output[outputOffset + 3] = imageData[outputOffset + 3];
            }
        }

        return output;
    }

    static _renderTwoColorBlend(quantization, plan) {
        const { width, height, pixelIndices, palette, imageData } = quantization;
        const output = new Uint8ClampedArray(width * height * 4);
        const replacementByIndex = plan.replacementByIndex;

        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                const pixelIndex = (y * width) + x;
                const outputOffset = pixelIndex * 4;
                const sourcePaletteIndex = pixelIndices[pixelIndex];
                const replacement = replacementByIndex[sourcePaletteIndex];
                const rgb = replacement ? replacement.mixedRgb : palette[sourcePaletteIndex];

                output[outputOffset] = rgb[0];
                output[outputOffset + 1] = rgb[1];
                output[outputOffset + 2] = rgb[2];
                output[outputOffset + 3] = imageData[outputOffset + 3];
            }
        }

        return output;
    }

    static _pixelNoise(x, y, seed = 0) {
        let hash = (
            Math.imul((x + 1) ^ 0x9e3779b9, 0x85ebca6b) ^
            Math.imul((y + 1) ^ 0xc2b2ae35, 0x27d4eb2d) ^
            Math.imul((seed + 1) ^ 0x165667b1, 0x1b873593)
        ) >>> 0;

        hash ^= hash >>> 15;
        hash = Math.imul(hash, 0x85ebca6b) >>> 0;
        hash ^= hash >>> 13;
        hash = Math.imul(hash, 0xc2b2ae35) >>> 0;
        hash ^= hash >>> 16;

        return (hash >>> 0) / 0x100000000;
    }

    static _findTopColors(imageData, k) {
        const width = imageData.width;
        const height = imageData.height;
        const sampleWidth = this.TOP_COLOR_SAMPLE_SIZE;
        const sampleHeight = this.TOP_COLOR_SAMPLE_SIZE;
        const colorCount = new Map();
        const smallCanvas = this._createCanvas(sampleWidth, sampleHeight);
        const smallContext = smallCanvas.getContext('2d', { willReadFrequently: true });
        const fullCanvas = this._createCanvas(width, height);
        const fullContext = fullCanvas.getContext('2d');

        if (!smallContext || !fullContext) {
            throw new Error('A 2D canvas context is required for K-means preprocessing.');
        }

        fullContext.putImageData(imageData, 0, 0);
        smallContext.imageSmoothingEnabled = false;
        smallContext.drawImage(fullCanvas, 0, 0, width, height, 0, 0, sampleWidth, sampleHeight);

        const pixels = new Uint32Array(smallContext.getImageData(0, 0, sampleWidth, sampleHeight).data.buffer);
        for (let pixelIndex = 0; pixelIndex < pixels.length; pixelIndex += 1) {
            const color = pixels[pixelIndex];
            colorCount.set(color, (colorCount.get(color) || 0) + 1);
        }

        const uniqueColors = [...colorCount.entries()]
            .sort((a, b) => b[1] - a[1])
            .map(([color]) => color);
        const limitedK = Math.max(1, Math.min(k, uniqueColors.length));
        const topColors = uniqueColors.slice(0, limitedK).map(color => [
            color & 0xff,
            (color >> 8) & 0xff,
            (color >> 16) & 0xff
        ]);

        return {
            topColors,
            sampleCount: sampleWidth * sampleHeight,
            uniqueSampleColors: uniqueColors.length
        };
    }

    static async _runKMeans(imageData, initialCentroids, onProgress) {
        if (initialCentroids.length === 0) {
            return { palette: [[0, 0, 0]], iterations: 0 };
        }

        const width = imageData.width;
        const height = imageData.height;
        const k = initialCentroids.length;
        const centroidR = new Float64Array(k);
        const centroidG = new Float64Array(k);
        const centroidB = new Float64Array(k);
        const nextR = new Float64Array(k);
        const nextG = new Float64Array(k);
        const nextB = new Float64Array(k);
        const sumR = new Float64Array(k);
        const sumG = new Float64Array(k);
        const sumB = new Float64Array(k);
        const clusterCounts = new Uint32Array(k);
        const pixels = imageData.data;
        const totalPixels = width * height;

        for (let centroidIndex = 0; centroidIndex < k; centroidIndex += 1) {
            centroidR[centroidIndex] = initialCentroids[centroidIndex][0];
            centroidG[centroidIndex] = initialCentroids[centroidIndex][1];
            centroidB[centroidIndex] = initialCentroids[centroidIndex][2];
        }

        let converged = false;
        let iterations = 0;

        while (!converged && iterations < this.MAX_KMEANS_ITERATIONS) {
            iterations += 1;

            sumR.fill(0);
            sumG.fill(0);
            sumB.fill(0);
            clusterCounts.fill(0);

            for (let y = 0; y < height; y += 1) {
                let offset = y * width * 4;
                for (let x = 0; x < width; x += 1, offset += 4) {
                    const red = pixels[offset];
                    const green = pixels[offset + 1];
                    const blue = pixels[offset + 2];
                    let minDistance = Infinity;
                    let closestCentroid = 0;

                    for (let centroidIndex = 0; centroidIndex < k; centroidIndex += 1) {
                        const distance = this._colorDifferenceEx(
                            red,
                            green,
                            blue,
                            centroidR[centroidIndex],
                            centroidG[centroidIndex],
                            centroidB[centroidIndex]
                        );
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestCentroid = centroidIndex;
                        }
                    }

                    sumR[closestCentroid] += red;
                    sumG[closestCentroid] += green;
                    sumB[closestCentroid] += blue;
                    clusterCounts[closestCentroid] += 1;
                }

                if (y % this.PROGRESS_YIELD_INTERVAL === 0 || y === height - 1) {
                    const progress = 20 + Math.round((((iterations - 1) + ((y + 1) / height)) / this.MAX_KMEANS_ITERATIONS) * 58);
                    await onProgress(
                        Math.min(78, progress),
                        `Refining centroids (${iterations}/${this.MAX_KMEANS_ITERATIONS}, row ${y + 1}/${height})...`
                    );
                }
            }

            for (let centroidIndex = 0; centroidIndex < k; centroidIndex += 1) {
                if (clusterCounts[centroidIndex] > 0) {
                    nextR[centroidIndex] = Math.floor(sumR[centroidIndex] / clusterCounts[centroidIndex]);
                    nextG[centroidIndex] = Math.floor(sumG[centroidIndex] / clusterCounts[centroidIndex]);
                    nextB[centroidIndex] = Math.floor(sumB[centroidIndex] / clusterCounts[centroidIndex]);
                } else {
                    nextR[centroidIndex] = centroidR[centroidIndex];
                    nextG[centroidIndex] = centroidG[centroidIndex];
                    nextB[centroidIndex] = centroidB[centroidIndex];
                }
            }

            converged = true;
            for (let centroidIndex = 0; centroidIndex < k; centroidIndex += 1) {
                if (
                    Math.abs(centroidR[centroidIndex] - nextR[centroidIndex]) +
                    Math.abs(centroidG[centroidIndex] - nextG[centroidIndex]) +
                    Math.abs(centroidB[centroidIndex] - nextB[centroidIndex]) >= 1
                ) {
                    converged = false;
                }
            }

            for (let centroidIndex = 0; centroidIndex < k; centroidIndex += 1) {
                centroidR[centroidIndex] = nextR[centroidIndex];
                centroidG[centroidIndex] = nextG[centroidIndex];
                centroidB[centroidIndex] = nextB[centroidIndex];
            }
        }

        const palette = new Array(k);
        for (let centroidIndex = 0; centroidIndex < k; centroidIndex += 1) {
            palette[centroidIndex] = [
                Math.round(centroidR[centroidIndex]),
                Math.round(centroidG[centroidIndex]),
                Math.round(centroidB[centroidIndex])
            ];
        }

        return {
            palette,
            iterations
        };
    }

    static async _applyPalette(imageData, palette, onProgress) {
        const { width, height, data } = imageData;
        const output = new Uint8ClampedArray(data.length);
        const pixelIndices = new Uint8Array(width * height);
        const usage = new Array(palette.length).fill(0);
        const paletteLength = palette.length;
        const paletteR = new Uint8ClampedArray(paletteLength);
        const paletteG = new Uint8ClampedArray(paletteLength);
        const paletteB = new Uint8ClampedArray(paletteLength);

        for (let paletteIndex = 0; paletteIndex < paletteLength; paletteIndex += 1) {
            paletteR[paletteIndex] = palette[paletteIndex][0];
            paletteG[paletteIndex] = palette[paletteIndex][1];
            paletteB[paletteIndex] = palette[paletteIndex][2];
        }

        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                const pixelIndex = (y * width) + x;
                const offset = pixelIndex * 4;
                const colorIndex = this._nearestPaletteIndex(
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    paletteR,
                    paletteG,
                    paletteB
                );

                output[offset] = paletteR[colorIndex];
                output[offset + 1] = paletteG[colorIndex];
                output[offset + 2] = paletteB[colorIndex];
                output[offset + 3] = data[offset + 3];
                pixelIndices[pixelIndex] = colorIndex;
                usage[colorIndex] += 1;
            }

            if (y % this.PROGRESS_YIELD_INTERVAL === 0 || y === height - 1) {
                const progress = 82 + Math.round(((y + 1) / height) * 16);
                await onProgress(progress, `Painting quantized pixels (${y + 1}/${height})...`);
            }
        }

        const totalPixels = width * height;
        const colorUsage = palette
            .map((rgb, index) => ({
                index,
                rgb,
                hex: this._toHex(rgb),
                count: usage[index],
                percent: (usage[index] / totalPixels) * 100
            }))
            .filter(entry => entry.count > 0)
            .sort((a, b) => b.count - a.count);

        return {
            output,
            pixelIndices,
            colorUsage
        };
    }

    static _nearestPaletteIndex(red, green, blue, paletteR, paletteG, paletteB) {
        let bestIndex = 0;
        let bestDistance = Infinity;

        for (let index = 0; index < paletteR.length; index += 1) {
            const distance = this._euclideanDistance(
                red,
                green,
                blue,
                paletteR[index],
                paletteG[index],
                paletteB[index]
            );
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = index;
            }
        }

        return bestIndex;
    }

    static _mixOklch(colorA, colorB, ratioA) {
        const ratioB = 1 - ratioA;
        const hue = this._interpolateHue(colorA, colorB, ratioA);

        return {
            l: (colorA.l * ratioA) + (colorB.l * ratioB),
            c: (colorA.c * ratioA) + (colorB.c * ratioB),
            h: hue
        };
    }

    static _interpolateHue(colorA, colorB, ratioA) {
        if (colorA.c <= this.ACHROMATIC_EPSILON && colorB.c <= this.ACHROMATIC_EPSILON) {
            return 0;
        }

        if (colorA.c <= this.ACHROMATIC_EPSILON) {
            return colorB.h;
        }

        if (colorB.c <= this.ACHROMATIC_EPSILON) {
            return colorA.h;
        }

        let delta = colorB.h - colorA.h;
        if (delta > Math.PI) {
            delta -= Math.PI * 2;
        } else if (delta < -Math.PI) {
            delta += Math.PI * 2;
        }

        return this._normalizeHue(colorA.h + (delta * (1 - ratioA)));
    }

    static _normalizeHue(hue) {
        const turn = Math.PI * 2;
        return ((hue % turn) + turn) % turn;
    }

    static _rgbToOklab(red, green, blue) {
        const linearRed = this._srgbChannelToLinear(red / 255);
        const linearGreen = this._srgbChannelToLinear(green / 255);
        const linearBlue = this._srgbChannelToLinear(blue / 255);

        const l = Math.cbrt((0.4122214708 * linearRed) + (0.5363325363 * linearGreen) + (0.0514459929 * linearBlue));
        const m = Math.cbrt((0.2119034982 * linearRed) + (0.6806995451 * linearGreen) + (0.1073969566 * linearBlue));
        const s = Math.cbrt((0.0883024619 * linearRed) + (0.2817188376 * linearGreen) + (0.6299787005 * linearBlue));

        return {
            l: (0.2104542553 * l) + (0.7936177850 * m) - (0.0040720468 * s),
            a: (1.9779984951 * l) - (2.4285922050 * m) + (0.4505937099 * s),
            b: (0.0259040371 * l) + (0.7827717662 * m) - (0.8086757660 * s)
        };
    }

    static _oklabToOklch(oklab) {
        return {
            l: oklab.l,
            c: Math.sqrt((oklab.a * oklab.a) + (oklab.b * oklab.b)),
            h: this._normalizeHue(Math.atan2(oklab.b, oklab.a))
        };
    }

    static _oklchToOklab(oklch) {
        return {
            l: oklch.l,
            a: oklch.c * Math.cos(oklch.h),
            b: oklch.c * Math.sin(oklch.h)
        };
    }

    static _oklabDistance(colorA, colorB) {
        const deltaL = colorA.l - colorB.l;
        const deltaA = colorA.a - colorB.a;
        const deltaB = colorA.b - colorB.b;
        return Math.sqrt((deltaL * deltaL) + (deltaA * deltaA) + (deltaB * deltaB));
    }

    static _oklabDistanceSquared(colorA, colorB) {
        const deltaL = colorA.l - colorB.l;
        const deltaA = colorA.a - colorB.a;
        const deltaB = colorA.b - colorB.b;
        return (deltaL * deltaL) + (deltaA * deltaA) + (deltaB * deltaB);
    }

    static _oklabToRgb(oklab) {
        const l = oklab.l + (0.3963377774 * oklab.a) + (0.2158037573 * oklab.b);
        const m = oklab.l - (0.1055613458 * oklab.a) - (0.0638541728 * oklab.b);
        const s = oklab.l - (0.0894841775 * oklab.a) - (1.2914855480 * oklab.b);

        const linearL = l * l * l;
        const linearM = m * m * m;
        const linearS = s * s * s;

        const red = (4.0767416621 * linearL) - (3.3077115913 * linearM) + (0.2309699292 * linearS);
        const green = (-1.2684380046 * linearL) + (2.6097574011 * linearM) - (0.3413193965 * linearS);
        const blue = (-0.0041960863 * linearL) - (0.7034186147 * linearM) + (1.7076147010 * linearS);

        return [
            this._linearToSrgbChannel(red),
            this._linearToSrgbChannel(green),
            this._linearToSrgbChannel(blue)
        ];
    }

    static _srgbChannelToLinear(channel) {
        if (channel <= 0.04045) {
            return channel / 12.92;
        }
        return Math.pow((channel + 0.055) / 1.055, 2.4);
    }

    static _linearToSrgbChannel(channel) {
        const clamped = Math.min(1, Math.max(0, channel));
        if (clamped <= 0.0031308) {
            return Math.round(clamped * 12.92 * 255);
        }
        return Math.round(((1.055 * Math.pow(clamped, 1 / 2.4)) - 0.055) * 255);
    }

    static _euclideanDistance(redA, greenA, blueA, redB, greenB, blueB) {
        const red = redA - redB;
        const green = greenA - greenB;
        const blue = blueA - blueB;
        return Math.sqrt((red * red) + (green * green) + (blue * blue));
    }

    static _colorDifferenceEx(redA, greenA, blueA, redB, greenB, blueB) {
        const redMean = (redA + redB) / 2;
        const redDelta = redA - redB;
        const greenDelta = greenA - greenB;
        const blueDelta = blueA - blueB;

        return Math.floor(
            (((512 + redMean) * redDelta * redDelta) >> 8) +
            (4 * greenDelta * greenDelta) +
            (((767 - redMean) * blueDelta * blueDelta) >> 8)
        );
    }

    static _toHex(rgb) {
        return `#${rgb.map(value => value.toString(16).padStart(2, '0')).join('')}`.toUpperCase();
    }

    static _createCanvas(width, height) {
        if (typeof OffscreenCanvas !== 'undefined') {
            return new OffscreenCanvas(width, height);
        }

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        return canvas;
    }
}

if (typeof window !== 'undefined') {
    window.ColorQuantizer = ColorQuantizer;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = ColorQuantizer;
}
