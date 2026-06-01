(function () {
  const IMAGE_EVENT = 'color-anything:image-selected';
  const REPRESENTATIVE_COLOR_MANIFEST = 'assets/sample-representative-colors.json';
  const DEFAULT_REPRESENTATIVE_COLOR_COUNT = 7;
  const FIXED_COLORS = [[0, 0, 0], [255, 255, 255]];
  const LCH_LIGHTNESS_LIMIT = 0.07;
  const LCH_DRIFT_LIMIT = 0.09;
  const LCH_NEUTRAL_CHROMA_LIMIT = 0.015;
  const LCH_CHROMA_LIMIT_FLOOR = 0.030;
  const LCH_CHROMA_LIMIT_RATIO = 0.40;
  const LCH_CHROMA_LIMIT_CEILING = 0.060;
  const DEFAULTS = {
    overlayEnabled: false,
    mergeEnabled: true,
    mergeThresholdPercent: 0.5,
    interpolationEnabled: true,
    interpolationSteps: 5
  };

  const originalCanvas = document.getElementById('caOriginalCanvas');
  const resultCanvas = document.getElementById('caResultCanvas');
  // These legacy meta/status elements are null in the new sidebar layout — we use no-op proxies
  const _noopEl = { textContent: '', innerHTML: '', disabled: false, style: {}, addEventListener: () => {} };
  const originalMeta = document.getElementById('caOriginalMeta') || _noopEl;
  const resultMeta = document.getElementById('caResultMeta') || _noopEl;
  const representativeHint = document.getElementById('caRepHint');
  const swatches = document.getElementById('caSwatches');
  const paletteStrip = document.getElementById('caPaletteStrip') || _noopEl;
  const lchEditor = document.getElementById('caLchEditor');
  const lchPreview = document.getElementById('caLchPreview');
  const lchPlane = document.getElementById('caLchPlane');
  const lchPlaneMarker = document.getElementById('caLchPlaneMarker');
  const lchPlaneLightnessRange = document.getElementById('caLchPlaneLightnessRange');
  const lchPlaneChromaRange = document.getElementById('caLchPlaneChromaRange');
  const lchLightnessValue = document.getElementById('caLchLightnessValue');
  const lchChromaValue = document.getElementById('caLchChromaValue');
  const lchHueSlider = document.getElementById('caLchHueSlider');
  const lchHueValue = document.getElementById('caLchHueValue');
  const lchLightness = document.getElementById('caLchLightness');
  const lchChroma = document.getElementById('caLchChroma');
  const lchClose = document.getElementById('caLchClose');
  const magnifier = document.getElementById('caMagnifier');
  const magnifierCanvas = document.getElementById('caMagnifierCanvas');
  const magnifierColor = document.getElementById('caMagnifierColor');
  const overlayToggle = document.getElementById('caOverlayToggle');
  const mergeToggle = document.getElementById('caMergeToggle');
  const mergeSlider = document.getElementById('caMergeSlider');
  const mergeValue = document.getElementById('caMergeValue');
  const interpolationSlider = document.getElementById('caInterpolationSlider');
  const interpolationValue = document.getElementById('caInterpolationValue');
  const saveButton = document.getElementById('caSaveBtn') || _noopEl;
  const usageTitle = document.getElementById('caUsageTitle') || _noopEl;
  const usageList = document.getElementById('caUsageList') || _noopEl;

  if (!originalCanvas || !resultCanvas) {
    return;
  }

  const originalCtx = originalCanvas.getContext('2d', { willReadFrequently: true });
  const resultCtx = resultCanvas.getContext('2d');
  const magnifierCtx = magnifierCanvas ? magnifierCanvas.getContext('2d') : null;
  const shared = window.ColorAnythingShared || (window.ColorAnythingShared = {});
  const MAG_PX = 140;
  const SRC_PX = 20;

  let currentImageData = null;
  let previewImageData = null;
  let currentImageSrc = '';
  let currentImageLabel = '';
  let representativeColors = [];
  let committedRepresentativeColors = [];
  let lastDetectionStats = null;
  let lastQuantization = null;
  let lastRenderResult = null;
  let activeImageLoadId = 0;
  let activeRenderId = 0;
  let rerenderTimer = null;
  let lchEditorTarget = -1;
  let lchEditorLocked = null;
  let lchEditorState = null;
  let lchPlanePointerId = null;
  let originalMetaBaseText = 'Select a shared source image';
  let representativeColorManifestPromise = null;

  function resolveImageRegion(image, label = '') {
    if (typeof shared.resolveImageRegion === 'function') {
      return shared.resolveImageRegion(image, label);
    }

    return {
      sourceX: 0,
      sourceY: 0,
      sourceWidth: image.naturalWidth,
      sourceHeight: image.naturalHeight,
      width: image.naturalWidth,
      height: image.naturalHeight,
      mode: 'original size'
    };
  }

  function normalizeInterpolationSteps(value) {
    return Math.max(1, Math.min(20, Math.round(Number(value) || DEFAULTS.interpolationSteps)));
  }

  function normalizeMergeThresholdPercent(value) {
    return Math.max(0, Math.min(5, Number(value) || DEFAULTS.mergeThresholdPercent));
  }

  function clampRgb(color) {
    return color.map(value => Math.max(0, Math.min(255, Math.round(value))));
  }

  function clamp(value, min, max) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return min;
    }
    return Math.max(min, Math.min(max, number));
  }

  function rgbToHex(color) {
    return `#${clampRgb(color).map(value => value.toString(16).padStart(2, '0')).join('')}`.toUpperCase();
  }

  function hexToRgb(hex) {
    const parsed = parseInt(hex.slice(1), 16);
    return [(parsed >> 16) & 255, (parsed >> 8) & 255, parsed & 255];
  }

  function clampHueDegrees(value) {
    return clamp(value, 0, 360);
  }

  function normalizeHueRadians(radians) {
    const turn = Math.PI * 2;
    return ((radians % turn) + turn) % turn;
  }

  function degreesToRadians(degrees) {
    return normalizeHueRadians(clampHueDegrees(degrees) * Math.PI / 180);
  }

  function radiansToDegrees(radians) {
    return normalizeHueRadians(radians) * 180 / Math.PI;
  }

  function rgbToLch(color) {
    if (!window.ColorQuantizer?._rgbToOklab || !window.ColorQuantizer?._oklabToOklch) {
      return { l: 0, c: 0, h: 0 };
    }

    return window.ColorQuantizer._oklabToOklch(
      window.ColorQuantizer._rgbToOklab(color[0], color[1], color[2])
    );
  }

  function lchToRgb(lch) {
    if (!window.ColorQuantizer?._oklchToOklab || !window.ColorQuantizer?._oklabToRgb) {
      return [0, 0, 0];
    }

    return clampRgb(
      window.ColorQuantizer._oklabToRgb(
        window.ColorQuantizer._oklchToOklab({
          l: lch.l,
          c: lch.c,
          h: normalizeHueRadians(lch.h)
        })
      )
    );
  }

  function buildLchHueGradient(lightness, chroma) {
    const stops = [];
    for (let hue = 0; hue <= 360; hue += 12) {
      const color = lchToRgb({ l: lightness, c: chroma, h: degreesToRadians(hue) });
      stops.push(`rgb(${color[0]},${color[1]},${color[2]}) ${((hue / 360) * 100).toFixed(2)}%`);
    }
    return `linear-gradient(90deg, ${stops.join(', ')})`;
  }

  function getLchChromaLimit(chroma) {
    if (chroma < 0.02) {
      return LCH_NEUTRAL_CHROMA_LIMIT;
    }
    return Math.min(
      LCH_CHROMA_LIMIT_CEILING,
      Math.max(LCH_CHROMA_LIMIT_FLOOR, chroma * LCH_CHROMA_LIMIT_RATIO)
    );
  }

  function buildLchEditorBounds(lch) {
    const chromaLimit = getLchChromaLimit(lch.c);
    return {
      lMin: clamp(lch.l - LCH_LIGHTNESS_LIMIT, 0, 1),
      lMax: clamp(lch.l + LCH_LIGHTNESS_LIMIT, 0, 1),
      cMin: Math.max(0, lch.c - chromaLimit),
      cMax: lch.c + chromaLimit
    };
  }

  function constrainLchEdit(lightness, chroma) {
    if (!lchEditorLocked) {
      return { l: 0, c: 0 };
    }

    let nextL = clamp(lightness, lchEditorLocked.lMin, lchEditorLocked.lMax);
    let nextC = clamp(chroma, lchEditorLocked.cMin, lchEditorLocked.cMax);
    const deltaL = nextL - lchEditorLocked.l;
    const drift = Math.abs(deltaL);

    if (drift > LCH_DRIFT_LIMIT) {
      const scale = LCH_DRIFT_LIMIT / drift;
      nextL = lchEditorLocked.l + (deltaL * scale);
    }

    return {
      l: clamp(nextL, lchEditorLocked.lMin, lchEditorLocked.lMax),
      c: clamp(nextC, lchEditorLocked.cMin, lchEditorLocked.cMax)
    };
  }

  function getLchPlaneGeometry() {
    if (!lchPlane) {
      return { width: 1, height: 1 };
    }
    return {
      width: lchPlane.width || 1,
      height: lchPlane.height || 1
    };
  }

  function lcToPlanePoint(lightness, chroma) {
    if (!lchEditorLocked) {
      return { x: 0, y: 0 };
    }

    const rect = lchPlane?.getBoundingClientRect();
    const width = rect?.width || 1;
    const height = rect?.height || 1;
    const chromaSpan = Math.max(lchEditorLocked.cMax - lchEditorLocked.cMin, 1e-6);
    const lightnessSpan = Math.max(lchEditorLocked.lMax - lchEditorLocked.lMin, 1e-6);
    const x = ((chroma - lchEditorLocked.cMin) / chromaSpan) * width;
    const y = ((lchEditorLocked.lMax - lightness) / lightnessSpan) * height;
    return {
      x: clamp(x, 0, width),
      y: clamp(y, 0, height)
    };
  }

  function planePointToLc(clientX, clientY) {
    if (!lchPlane || !lchEditorLocked) {
      return { l: 0, c: 0 };
    }

    const rect = lchPlane.getBoundingClientRect();
    const px = clamp(clientX - rect.left, 0, rect.width);
    const py = clamp(clientY - rect.top, 0, rect.height);
    const nx = rect.width > 0 ? px / rect.width : 0;
    const ny = rect.height > 0 ? py / rect.height : 0;

    return constrainLchEdit(
      lchEditorLocked.lMax - (ny * (lchEditorLocked.lMax - lchEditorLocked.lMin)),
      lchEditorLocked.cMin + (nx * (lchEditorLocked.cMax - lchEditorLocked.cMin))
    );
  }

  function renderLchPlane(hueDegrees) {
    if (!lchPlane || !lchEditorLocked) {
      return;
    }

    const context = lchPlane.getContext('2d');
    if (!context) {
      return;
    }

    const { width, height } = getLchPlaneGeometry();
    const image = context.createImageData(width, height);
    const data = image.data;
    const hue = degreesToRadians(hueDegrees);
    const lightnessSpan = Math.max(lchEditorLocked.lMax - lchEditorLocked.lMin, 1e-6);
    const chromaSpan = Math.max(lchEditorLocked.cMax - lchEditorLocked.cMin, 1e-6);

    for (let y = 0; y < height; y += 1) {
      const lightness = lchEditorLocked.lMax - ((y / Math.max(height - 1, 1)) * lightnessSpan);
      for (let x = 0; x < width; x += 1) {
        const chroma = lchEditorLocked.cMin + ((x / Math.max(width - 1, 1)) * chromaSpan);
        const color = lchToRgb({ l: lightness, c: chroma, h: hue });
        const offset = ((y * width) + x) * 4;
        data[offset] = color[0];
        data[offset + 1] = color[1];
        data[offset + 2] = color[2];
        data[offset + 3] = 255;
      }
    }

    context.putImageData(image, 0, 0);
  }

  function hideLchEditor() {
    lchEditorTarget = -1;
    lchEditorLocked = null;
    lchEditorState = null;
    if (lchEditor) {
      lchEditor.hidden = true;
    }
  }

  function renderLchEditorView(lch, previewColor = null) {
    if (!lchEditor || !lchEditorLocked) {
      return;
    }

    const hue = clampHueDegrees(lch.h);
    const constrained = constrainLchEdit(lch.l, lch.c);
    const color = previewColor || lchToRgb({
      l: constrained.l,
      c: constrained.c,
      h: degreesToRadians(hue)
    });

    if (lchPreview) {
      lchPreview.style.background = `rgb(${color[0]},${color[1]},${color[2]})`;
    }
    if (lchLightness) {
      lchLightness.textContent = (lchEditorLocked.l * 100).toFixed(1);
    }
    if (lchChroma) {
      lchChroma.textContent = lchEditorLocked.c.toFixed(3);
    }
    if (lchLightnessValue) {
      lchLightnessValue.textContent = `L ${(constrained.l * 100).toFixed(1)}`;
    }
    if (lchChromaValue) {
      lchChromaValue.textContent = `C ${constrained.c.toFixed(3)}`;
    }
    if (lchPlaneLightnessRange) {
      lchPlaneLightnessRange.textContent = `L ${(lchEditorLocked.lMin * 100).toFixed(1)} - ${(lchEditorLocked.lMax * 100).toFixed(1)}`;
    }
    if (lchPlaneChromaRange) {
      lchPlaneChromaRange.textContent = `C ${lchEditorLocked.cMin.toFixed(3)} - ${lchEditorLocked.cMax.toFixed(3)}`;
    }
    if (lchHueValue) {
      lchHueValue.textContent = `${Math.round(hue)}°`;
    }
    if (lchHueSlider) {
      lchHueSlider.value = String(Math.round(hue));
    }
    if (lchHueSlider) {
      lchHueSlider.style.setProperty(
        '--lch-hue-gradient',
        buildLchHueGradient(constrained.l, constrained.c)
      );
    }
    renderLchPlane(hue);
    if (lchPlaneMarker) {
      const point = lcToPlanePoint(constrained.l, constrained.c);
      lchPlaneMarker.style.left = `${point.x}px`;
      lchPlaneMarker.style.top = `${point.y}px`;
      lchPlaneMarker.style.background = `rgb(${color[0]},${color[1]},${color[2]})`;
    }
  }

  function openLchEditor(representativeIndex) {
    if (
      !lchEditor ||
      !lchPlane ||
      !lchHueSlider ||
      representativeIndex < 0 ||
      representativeIndex >= representativeColors.length
    ) {
      return;
    }

    const currentColor = representativeColors[representativeIndex];
    const lch = rgbToLch(currentColor);
    const hueDegrees = radiansToDegrees(lch.h);
    lchEditorTarget = representativeIndex;
    lchEditorLocked = {
      l: lch.l,
      c: lch.c,
      h: hueDegrees,
      ...buildLchEditorBounds(lch)
    };
    lchEditorState = {
      l: lch.l,
      c: lch.c,
      h: hueDegrees
    };
    lchEditor.hidden = false;
    renderLchEditorView(lchEditorState, currentColor);
    renderRepresentativeSwatches();
    window.requestAnimationFrame(() => lchHueSlider.focus());
  }

  function handleLchEditorInput() {
    if (!lchEditorLocked || lchEditorTarget < 0 || lchEditorTarget >= representativeColors.length) {
      return;
    }

    const hueDegrees = clampHueDegrees(lchHueSlider?.value ?? lchEditorLocked.h);
    const constrained = constrainLchEdit(lchEditorState?.l ?? lchEditorLocked.l, lchEditorState?.c ?? lchEditorLocked.c);
    const nextColor = lchToRgb({
      l: constrained.l,
      c: constrained.c,
      h: degreesToRadians(hueDegrees)
    });
    lchEditorState = { l: constrained.l, c: constrained.c, h: hueDegrees };
    representativeColors[lchEditorTarget] = nextColor;
    lastQuantization = buildQuantizationModel();
    renderLchEditorView(lchEditorState, nextColor);
    renderRepresentativeSwatches();
    applyRepresentativeEditPreview();
  }

  function handleLchPlanePointer(event) {
    if (!lchPlane || !lchEditorLocked) {
      return;
    }

    const nextLc = planePointToLc(event.clientX, event.clientY);
    lchEditorState = {
      l: nextLc.l,
      c: nextLc.c,
      h: clampHueDegrees(lchHueSlider?.value ?? lchEditorLocked.h)
    };
    handleLchEditorInput();
  }

  function getSampleKey(label = '') {
    const match = String(label).match(/(Sample\d+)/i);
    return match ? `${match[1].charAt(0).toUpperCase()}${match[1].slice(1)}` : '';
  }

  async function loadRepresentativeColorManifest() {
    if (!representativeColorManifestPromise) {
      representativeColorManifestPromise = fetch(REPRESENTATIVE_COLOR_MANIFEST)
        .then(response => {
          if (!response.ok) {
            throw new Error(`Could not load ${REPRESENTATIVE_COLOR_MANIFEST}`);
          }
          return response.json();
        })
        .catch(error => {
          representativeColorManifestPromise = null;
          throw error;
        });
    }

    return representativeColorManifestPromise;
  }

  async function getRepresentativeColorsForSelection(label) {
    const sampleKey = getSampleKey(label);
    if (!sampleKey) {
      return null;
    }

    const manifest = await loadRepresentativeColorManifest();
    const colors = manifest?.[sampleKey];
    if (!Array.isArray(colors) || colors.length === 0) {
      return null;
    }

    return sanitizeRepresentativeColors(colors.map(hexToRgb));
  }

  function sanitizeRepresentativeColors(colors) {
    const seen = new Set();
    const output = [];
    for (const color of colors) {
      if (!Array.isArray(color) || color.length < 3) {
        continue;
      }
      const normalized = clampRgb([color[0], color[1], color[2]]);
      const key = rgbToHex(normalized);
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      output.push(normalized);
    }
    return output;
  }

  function cloneRepresentativeColors(colors) {
    return colors.map(color => [...color]);
  }

  function buildPalette(colors = representativeColors) {
    return [FIXED_COLORS[0], ...colors, FIXED_COLORS[1]];
  }

  function buildQuantizationModelForImage(imageData, colors = representativeColors) {
    if (!imageData) {
      return null;
    }

    const palette = buildPalette(colors);
    return {
      width: imageData.width,
      height: imageData.height,
      palette,
      stats: {
        requestedColors: colors.length,
        centroidCount: palette.length,
        sampleCount: lastDetectionStats?.sampleCount || 0,
        uniqueSampleColors: lastDetectionStats?.uniqueSampleColors || 0,
        totalPixels: imageData.width * imageData.height,
        trainingPixelCount: lastDetectionStats?.trainingPixelCount || 0,
        trainingStep: lastDetectionStats?.trainingStep || 1,
        iterations: lastDetectionStats?.iterations || 0
      },
      elapsedMs: lastDetectionStats?.elapsedMs || 0
    };
  }

  function buildQuantizationModel() {
    return buildQuantizationModelForImage(currentImageData);
  }

  function updateControlLabels() {
    mergeValue.textContent = `${normalizeMergeThresholdPercent(mergeSlider.value).toFixed(1)}%`;
    interpolationValue.textContent = String(normalizeInterpolationSteps(interpolationSlider.value));
  }

  function getRenderAdjustmentSummary() {
    const adjustments = [];
    if (overlayToggle.checked) {
      adjustments.push('lightness overlay');
    }
    return adjustments.length ? ` · ${adjustments.join(' · ')}` : '';
  }

  function getRenderOptions() {
    return {
      overlayEnabled: overlayToggle.checked,
      mergeEnabled: mergeToggle.checked,
      mergeThreshold: normalizeMergeThresholdPercent(mergeSlider.value) / 100,
      interpolationEnabled: true,
      interpolationSteps: normalizeInterpolationSteps(interpolationSlider.value)
    };
  }

  function applySettings(settings) {
    const merged = { ...DEFAULTS, ...settings };
    overlayToggle.checked = merged.overlayEnabled === true;
    mergeToggle.checked = merged.mergeEnabled !== false;
    mergeSlider.value = String(normalizeMergeThresholdPercent(merged.mergeThresholdPercent));
    interpolationSlider.value = String(normalizeInterpolationSteps(merged.interpolationSteps));
    updateControlLabels();
  }

  function loadSettings() {
    applySettings(DEFAULTS);
  }

  function rebuildQuantizationModel() {
    lastQuantization = buildQuantizationModel();
    lastRenderResult = null;
  }

  function setOriginalProcessingLabel(message) {
    originalMeta.textContent = message;
  }

  function restoreOriginalMetaLabel() {
    originalMeta.textContent = originalMetaBaseText;
  }

  async function updateProgress(percent, text) {
    if (text) {
      resultMeta.textContent = text;
    }
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  function renderPaletteStrip(palette) {
    paletteStrip.innerHTML = palette
      .map(color => `<div class="palette-strip-color" style="background:rgb(${color[0]},${color[1]},${color[2]})"></div>`)
      .join('');
  }

  function buildSequentialDisplayPalette() {
    const anchors = buildPalette();
    if (anchors.length < 2 || !window.ColorQuantizer?.buildInterpolatedPalette) {
      return anchors;
    }

    const steps = normalizeInterpolationSteps(interpolationSlider.value);
    const ordered = [[...anchors[0]]];

    for (let index = 0; index < anchors.length - 1; index += 1) {
      const pairPalette = window.ColorQuantizer.buildInterpolatedPalette(
        [anchors[index], anchors[index + 1]],
        steps,
        true
      ).palette;
      ordered.push(...pairPalette.slice(2), [...anchors[index + 1]]);
    }

    return ordered;
  }

  function getDisplayPalette() {
    return buildSequentialDisplayPalette();
  }

  function renderRepresentativeSwatches() {
    if (lchEditorTarget >= representativeColors.length) {
      hideLchEditor();
    }

    swatches.innerHTML = '';
    const palette = buildPalette();
    representativeHint.textContent = representativeColors.length > 0
      ? 'click a swatch to adjust LCH in the picker, or click the original image to add another color'
      : 'click the original image to add representative colors';
    if (representativeHint) representativeHint.style.display = representativeColors.length === 0 ? '' : 'none';

    palette.forEach((color, index) => {
      const isFixed = index === 0 || index === palette.length - 1;
      const representativeIndex = index - 1;
      const wrap = document.createElement('div');
      wrap.className = 'ca-swatch-wrap';
      const swatch = document.createElement('div');
      swatch.className = `ca-swatch${isFixed ? ' fixed' : ''}`;
      swatch.style.background = `rgb(${color[0]},${color[1]},${color[2]})`;
      swatch.title = isFixed
        ? (index === 0 ? 'Black (fixed)' : 'White (fixed)')
        : `Representative color ${representativeIndex + 1}`;
      if (!isFixed && representativeIndex === lchEditorTarget) {
        swatch.classList.add('active');
      }

      if (!isFixed) {
        swatch.addEventListener('click', () => {
          openLchEditor(representativeIndex);
        });

        const removeButton = document.createElement('button');
        removeButton.className = 'ca-swatch-remove';
        removeButton.title = 'Remove';
        removeButton.textContent = '×';
        removeButton.addEventListener('click', event => {
          event.stopPropagation();
          if (representativeIndex === lchEditorTarget) {
            hideLchEditor();
          } else if (representativeIndex < lchEditorTarget) {
            lchEditorTarget -= 1;
          }
          representativeColors.splice(representativeIndex, 1);
          representativeColors = sanitizeRepresentativeColors(representativeColors);
          rebuildQuantizationModel();
          renderRepresentativeSwatches();
          scheduleRerender();
        });
        wrap.appendChild(removeButton);
      }

      wrap.appendChild(swatch);
      swatches.appendChild(wrap);
    });

    renderPaletteStrip(getDisplayPalette());
  }

  function hideMagnifier() {
    if (magnifier) {
      magnifier.style.display = 'none';
    }
  }

  function buildUsageHtml(entries) {
    return entries.map(entry => {
      const swatchStyle = `background:rgb(${entry.color[0]},${entry.color[1]},${entry.color[2]})`;
      return `
        <div class="ca-usage-card">
          <div class="ca-usage-swatch" style="${swatchStyle}"></div>
          <div class="ca-usage-meta">
            <span class="ca-usage-kind is-${entry.kind}">${entry.kind}</span>
            <div class="ca-usage-pct">${entry.percent.toFixed(1)}%</div>
            <div class="ca-usage-hex">${entry.hex}</div>
          </div>
        </div>
      `;
    }).join('');
  }

  function renderUsagePreview(renderResult) {
    if (!renderResult) {
      usageTitle.textContent = 'Colors used';
      usageList.innerHTML = '';
      return;
    }
    const { stats } = renderResult;
    usageTitle.textContent = `Colors used — ${stats.representativeEntryCount} representative${stats.interpolatedEntryCount > 0 ? `, ${stats.interpolatedEntryCount} interpolated` : ''}`;
    usageList.innerHTML = buildUsageHtml(renderResult.usageEntries);
  }

  function clearResultState(message = 'Loading representative colors...') {
    lastQuantization = null;
    lastRenderResult = null;
    clearTimeout(rerenderTimer);
    resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    resultCanvas.width = originalCanvas.width;
    resultCanvas.height = originalCanvas.height;
    resultMeta.textContent = 'result pending';
    saveButton.disabled = true;
    renderUsagePreview(null);
  }

  async function renderCurrentOutput(showProgress = false) {
    if (!currentImageData || !lastQuantization) {
      return;
    }
    if (!Array.isArray(lastQuantization.palette) || lastQuantization.palette.length === 0) {
      resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
      resultMeta.textContent = 'add representative colors';
      saveButton.disabled = true;
      renderPaletteStrip([]);
      renderUsagePreview(null);
      return;
    }

    const renderId = ++activeRenderId;
    saveButton.disabled = true;
    if (showProgress) {
      setOriginalProcessingLabel('Processing source image...');
      resultMeta.textContent = 'waiting for output...';
    } else {
      resultMeta.textContent = 'updating preview...';
    }
    try {
      const renderResult = await window.ColorQuantizer.renderColorAnything(
        currentImageData,
        lastQuantization,
        getRenderOptions(),
        async (percent, text) => {
          if (renderId !== activeRenderId || !showProgress) {
            return;
          }
          await updateProgress(percent, text);
        }
      );

      if (renderId !== activeRenderId) {
        return;
      }

      lastRenderResult = renderResult;
      committedRepresentativeColors = cloneRepresentativeColors(representativeColors);
      resultCanvas.width = renderResult.width;
      resultCanvas.height = renderResult.height;
      resultCtx.putImageData(new ImageData(renderResult.imageData, renderResult.width, renderResult.height), 0, 0);
      if (window.ColorAnythingCA?.onRender) {
        const hexColors = [FIXED_COLORS[0], ...representativeColors, FIXED_COLORS[1]]
          .map(c => rgbToHex(c));
        const resultUrl = resultCanvas.toDataURL('image/png');
        window.ColorAnythingCA.onRender(resultUrl, hexColors, renderResult.usageEntries);
      }
      const { stats } = renderResult;
      renderPaletteStrip(renderResult.palette);
      restoreOriginalMetaLabel();
      resultMeta.textContent = `${stats.usedPaletteEntries} mapped entries · ${stats.representativeEntryCount} representative${stats.interpolatedEntryCount > 0 ? `, ${stats.interpolatedEntryCount} interpolated` : ''}${getRenderAdjustmentSummary()}`;
      saveButton.disabled = false;
      renderUsagePreview(renderResult);
    } catch (error) {
      restoreOriginalMetaLabel();
      renderUsagePreview(null);
      resultMeta.textContent = 'processing failed';
      saveButton.disabled = true;
    }
  }

  function scheduleRerender() {
    if (!currentImageData || !lastQuantization) {
      return;
    }

    clearTimeout(rerenderTimer);
    rerenderTimer = setTimeout(() => {
      renderCurrentOutput(false);
    }, 60);
  }

  function applyRepresentativeEditPreview() {
    if (!currentImageData || !lastQuantization || !lastRenderResult || !window.ColorQuantizer?.recolorWithExistingMapping) {
      scheduleRerender();
      return;
    }

    try {
      const renderResult = window.ColorQuantizer.recolorWithExistingMapping(
        currentImageData,
        lastQuantization,
        lastRenderResult,
        getRenderOptions()
      );
      lastRenderResult = renderResult;
      resultCanvas.width = renderResult.width;
      resultCanvas.height = renderResult.height;
      resultCtx.putImageData(new ImageData(renderResult.imageData, renderResult.width, renderResult.height), 0, 0);
      if (window.ColorAnythingCA?.onRender) {
        const hexColors = [FIXED_COLORS[0], ...representativeColors, FIXED_COLORS[1]]
          .map(c => rgbToHex(c));
        const resultUrl = resultCanvas.toDataURL('image/png');
        window.ColorAnythingCA.onRender(resultUrl, hexColors, renderResult.usageEntries);
      }
      const { stats } = renderResult;
      renderPaletteStrip(renderResult.palette);
      resultMeta.textContent = `${stats.usedPaletteEntries} mapped entries · ${stats.representativeEntryCount} representative${stats.interpolatedEntryCount > 0 ? `, ${stats.interpolatedEntryCount} interpolated` : ''}${getRenderAdjustmentSummary()}`;
      saveButton.disabled = false;
      renderUsagePreview(renderResult);
    } catch (error) {
      scheduleRerender();
    }
  }

  async function detectAndRender(label, showProgress = true) {
    if (!currentImageData) {
      return;
    }

    clearTimeout(rerenderTimer);
    saveButton.disabled = true;
    setOriginalProcessingLabel('Analyzing source image...');
    resultMeta.textContent = 'waiting for output...';

    try {
      const predefinedColors = await getRepresentativeColorsForSelection(label);
      if (predefinedColors) {
        representativeColors = predefinedColors;
        lastDetectionStats = {
          requestedColors: representativeColors.length,
          centroidCount: representativeColors.length + FIXED_COLORS.length,
          totalPixels: currentImageData.width * currentImageData.height,
          sampleCount: 0,
          uniqueSampleColors: 0,
          trainingPixelCount: 0,
          trainingStep: 1,
          iterations: 0,
          elapsedMs: 0
        };
      } else {
        const startedAt = performance.now();
        const detected = await window.ColorQuantizer.detectRepresentativeColors(
          currentImageData,
          DEFAULT_REPRESENTATIVE_COLOR_COUNT,
          updateProgress
        );
        representativeColors = sanitizeRepresentativeColors(detected.palette);
        lastDetectionStats = { ...detected.stats, elapsedMs: performance.now() - startedAt };
      }
      rebuildQuantizationModel();
      renderRepresentativeSwatches();
      await renderCurrentOutput(showProgress);
    } catch (error) {
      restoreOriginalMetaLabel();
      representativeColors = [];
      lastDetectionStats = null;
      rebuildQuantizationModel();
      renderRepresentativeSwatches();
      renderUsagePreview(null);
      resultMeta.textContent = 'processing failed';
    }
  }

  function canvasToBlob(canvas) {
    return new Promise((resolve, reject) => {
      canvas.toBlob(blob => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Could not encode the processed image.'));
        }
      }, 'image/png');
    });
  }

  async function saveResult() {
    if (!lastRenderResult || !currentImageSrc) {
      return;
    }

    const saveRenderId = ++activeRenderId;
    saveButton.disabled = true;
    setOriginalProcessingLabel('Loading original image for save...');
    resultMeta.textContent = 'rendering full-resolution output...';

    try {
      const image = await new Promise((resolve, reject) => {
        const nextImage = new Image();
        nextImage.onload = () => resolve(nextImage);
        nextImage.onerror = () => reject(new Error('Could not load the original image for saving.'));
        nextImage.src = currentImageSrc;
      });

      if (saveRenderId !== activeRenderId) {
        return;
      }

      const imageRegion = resolveImageRegion(image, currentImageLabel);
      const sourceCanvas = document.createElement('canvas');
      sourceCanvas.width = imageRegion.width;
      sourceCanvas.height = imageRegion.height;
      const sourceCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });
      if (!sourceCtx) {
        throw new Error('Could not create a full-resolution source context.');
      }

      sourceCtx.drawImage(
        image,
        imageRegion.sourceX,
        imageRegion.sourceY,
        imageRegion.sourceWidth,
        imageRegion.sourceHeight,
        0,
        0,
        imageRegion.width,
        imageRegion.height
      );
      const fullImageData = sourceCtx.getImageData(0, 0, imageRegion.width, imageRegion.height);
      const baseColors = committedRepresentativeColors.length > 0
        ? committedRepresentativeColors
        : representativeColors;
      const baseQuantization = buildQuantizationModelForImage(fullImageData, baseColors);
      const currentQuantization = buildQuantizationModelForImage(fullImageData, representativeColors);
      const baseRenderResult = await window.ColorQuantizer.renderColorAnything(
        fullImageData,
        baseQuantization,
        getRenderOptions(),
        async () => {}
      );

      if (saveRenderId !== activeRenderId) {
        return;
      }

      const finalRenderResult = window.ColorQuantizer.recolorWithExistingMapping(
        fullImageData,
        currentQuantization,
        baseRenderResult,
        getRenderOptions()
      );

      const exportCanvas = document.createElement('canvas');
      exportCanvas.width = finalRenderResult.width;
      exportCanvas.height = finalRenderResult.height;
      const exportCtx = exportCanvas.getContext('2d');
      if (!exportCtx) {
        throw new Error('Could not create an export canvas context.');
      }
      exportCtx.putImageData(new ImageData(finalRenderResult.imageData, finalRenderResult.width, finalRenderResult.height), 0, 0);

      const blob = await canvasToBlob(exportCanvas);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.download = 'kmeans-color-anything.png';
      link.href = url;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      restoreOriginalMetaLabel();
      resultMeta.textContent = `${lastRenderResult.stats.usedPaletteEntries} mapped entries · ${lastRenderResult.stats.representativeEntryCount} representative${lastRenderResult.stats.interpolatedEntryCount > 0 ? `, ${lastRenderResult.stats.interpolatedEntryCount} interpolated` : ''}${getRenderAdjustmentSummary()}`;
      saveButton.disabled = false;
    } catch (error) {
      restoreOriginalMetaLabel();
      resultMeta.textContent = 'save failed';
      saveButton.disabled = !lastRenderResult;
    }
  }

  async function loadSharedImage(src, label = src) {
    const loadId = ++activeImageLoadId;
    currentImageSrc = src;
    currentImageLabel = label;
    hideMagnifier();
    originalMeta.textContent = `Loading ${label}...`;
    resultMeta.textContent = 'loading image...';

    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = async () => {
        if (loadId !== activeImageLoadId) {
          resolve(false);
          return;
        }

        try {
          const imageRegion = resolveImageRegion(image, label);
          const sourceCanvas = document.createElement('canvas');
          sourceCanvas.width = imageRegion.width;
          sourceCanvas.height = imageRegion.height;
          const sourceCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });
          if (!sourceCtx) {
            throw new Error('Could not create source canvas context.');
          }

          sourceCtx.drawImage(
            image,
            imageRegion.sourceX,
            imageRegion.sourceY,
            imageRegion.sourceWidth,
            imageRegion.sourceHeight,
            0,
            0,
            imageRegion.width,
            imageRegion.height
          );
          currentImageData = sourceCtx.getImageData(0, 0, imageRegion.width, imageRegion.height);
          previewImageData = currentImageData;

          originalCanvas.width = imageRegion.width;
          originalCanvas.height = imageRegion.height;
          resultCanvas.width = imageRegion.width;
          resultCanvas.height = imageRegion.height;

          originalCtx.clearRect(0, 0, imageRegion.width, imageRegion.height);
          originalCtx.drawImage(
            image,
            imageRegion.sourceX,
            imageRegion.sourceY,
            imageRegion.sourceWidth,
            imageRegion.sourceHeight,
            0,
            0,
            imageRegion.width,
            imageRegion.height
          );

          representativeColors = [];
          committedRepresentativeColors = [];
          lastDetectionStats = null;
          rebuildQuantizationModel();
          renderRepresentativeSwatches();
          clearResultState('Detecting representative colors...');
          originalMetaBaseText = `${imageRegion.width} × ${imageRegion.height} px (${imageRegion.mode})${label ? ` • ${label}` : ''}`;
          restoreOriginalMetaLabel();

          if (window.ColorAnythingCA?.onImageLoaded) {
            const origUrl = originalCanvas.toDataURL('image/png');
            window.ColorAnythingCA.onImageLoaded(origUrl, label, imageRegion.width, imageRegion.height);
          }
          await detectAndRender(label, true);
          resolve(true);
        } catch (error) {
          currentImageData = null;
          previewImageData = null;
          representativeColors = [];
          committedRepresentativeColors = [];
          lastDetectionStats = null;
          rebuildQuantizationModel();
          renderRepresentativeSwatches();
          originalMetaBaseText = 'Could not load the selected image.';
          originalMeta.textContent = 'Could not load the selected image.';
          clearResultState('Image loading failed.');
          resultMeta.textContent = 'image load failed';
          reject(error);
        }
      };
      image.onerror = () => {
        currentImageData = null;
        previewImageData = null;
        representativeColors = [];
        committedRepresentativeColors = [];
        lastDetectionStats = null;
        rebuildQuantizationModel();
        renderRepresentativeSwatches();
        originalMetaBaseText = 'Could not load the selected image.';
        originalMeta.textContent = 'Could not load the selected image.';
        clearResultState('Image loading failed.');
        resultMeta.textContent = 'image load failed';
        reject(new Error('Could not load the selected image.'));
      };
      image.src = src;
    });
  }

  overlayToggle.addEventListener('change', () => {
    applyRepresentativeEditPreview();
  });

  mergeToggle.addEventListener('change', () => {
    scheduleRerender();
  });

  mergeSlider.addEventListener('input', () => {
    updateControlLabels();
    scheduleRerender();
  });

  interpolationSlider.addEventListener('input', () => {
    updateControlLabels();
    lastRenderResult = null;
    renderPaletteStrip(getDisplayPalette());
    scheduleRerender();
  });

  originalCanvas.addEventListener('click', event => {
    if (!currentImageData) {
      return;
    }
    const rect = originalCanvas.getBoundingClientRect();
    const scaleX = currentImageData.width / rect.width;
    const scaleY = currentImageData.height / rect.height;
    const x = Math.floor((event.clientX - rect.left) * scaleX);
    const y = Math.floor((event.clientY - rect.top) * scaleY);
    const offset = (y * currentImageData.width + x) * 4;
    const selectedColor = [
      currentImageData.data[offset],
      currentImageData.data[offset + 1],
      currentImageData.data[offset + 2]
    ];
    representativeColors = sanitizeRepresentativeColors([
      ...representativeColors,
      selectedColor
    ]);
    rebuildQuantizationModel();
    renderRepresentativeSwatches();
    scheduleRerender();
  });

  originalCanvas.addEventListener('mousemove', event => {
    if (!currentImageData || !previewImageData || !magnifier || !magnifierCtx || !magnifierColor) {
      return;
    }

    const rect = originalCanvas.getBoundingClientRect();
    const previewScaleX = originalCanvas.width / rect.width;
    const previewScaleY = originalCanvas.height / rect.height;
    const sourceScaleX = currentImageData.width / rect.width;
    const sourceScaleY = currentImageData.height / rect.height;
    const previewCx = Math.floor((event.clientX - rect.left) * previewScaleX);
    const previewCy = Math.floor((event.clientY - rect.top) * previewScaleY);
    const cx = Math.floor((event.clientX - rect.left) * sourceScaleX);
    const cy = Math.floor((event.clientY - rect.top) * sourceScaleY);

    const half = SRC_PX / 2;
    const sx = Math.max(0, Math.min(originalCanvas.width - SRC_PX, previewCx - half));
    const sy = Math.max(0, Math.min(originalCanvas.height - SRC_PX, previewCy - half));

    magnifierCtx.imageSmoothingEnabled = false;
    magnifierCtx.clearRect(0, 0, MAG_PX, MAG_PX);
    magnifierCtx.drawImage(originalCanvas, sx, sy, SRC_PX, SRC_PX, 0, 0, MAG_PX, MAG_PX);

    const mid = MAG_PX / 2;
    const gap = 5;
    magnifierCtx.strokeStyle = 'rgba(255,255,255,0.85)';
    magnifierCtx.lineWidth = 1;
    magnifierCtx.beginPath();
    magnifierCtx.moveTo(0, mid);
    magnifierCtx.lineTo(mid - gap, mid);
    magnifierCtx.moveTo(mid + gap, mid);
    magnifierCtx.lineTo(MAG_PX, mid);
    magnifierCtx.moveTo(mid, 0);
    magnifierCtx.lineTo(mid, mid - gap);
    magnifierCtx.moveTo(mid, mid + gap);
    magnifierCtx.lineTo(mid, MAG_PX);
    magnifierCtx.stroke();

    const pixelIndex = (cy * currentImageData.width + cx) * 4;
    magnifierColor.textContent = rgbToHex([
      currentImageData.data[pixelIndex],
      currentImageData.data[pixelIndex + 1],
      currentImageData.data[pixelIndex + 2]
    ]);

    const pad = 20;
    let left = event.clientX + pad;
    let top = event.clientY + pad;
    if (left + MAG_PX + 20 > window.innerWidth) {
      left = event.clientX - MAG_PX - pad;
    }
    if (top + MAG_PX + 30 > window.innerHeight) {
      top = event.clientY - MAG_PX - pad - 30;
    }

    magnifier.style.left = `${left}px`;
    magnifier.style.top = `${top}px`;
    magnifier.style.display = 'flex';
  });

  originalCanvas.addEventListener('mouseleave', hideMagnifier);

  if (lchPlane) {
    lchPlane.addEventListener('pointerdown', event => {
      lchPlanePointerId = event.pointerId;
      lchPlane.setPointerCapture(event.pointerId);
      handleLchPlanePointer(event);
    });
    lchPlane.addEventListener('pointermove', event => {
      if (lchPlanePointerId === event.pointerId) {
        handleLchPlanePointer(event);
      }
    });
    const releaseLchPlanePointer = event => {
      if (lchPlanePointerId === event.pointerId) {
        lchPlanePointerId = null;
      }
    };
    lchPlane.addEventListener('pointerup', releaseLchPlanePointer);
    lchPlane.addEventListener('pointercancel', releaseLchPlanePointer);
  }

  if (lchHueSlider) {
    lchHueSlider.addEventListener('input', handleLchEditorInput);
    lchHueSlider.addEventListener('change', handleLchEditorInput);
  }

  if (lchClose) {
    lchClose.addEventListener('click', () => {
      hideLchEditor();
      renderRepresentativeSwatches();
    });
  }

  saveButton.addEventListener('click', saveResult);

  loadSettings();
  renderRepresentativeSwatches();
  renderUsagePreview(null);
  resultMeta.textContent = 'result pending';

  // Public API + callbacks for the shell UI
  window.ColorAnythingCA = {
    onImageLoaded: null,  // (originalDataUrl, filename, w, h) => void
    onRender: null,       // (resultDataUrl, hexColors, entryCount) => void

    loadImage: loadSharedImage,
    getRepresentativeColors: () => representativeColors.map(c => rgbToHex(c)),
    setRepresentativeColors(hexList) {
      representativeColors = sanitizeRepresentativeColors(hexList.map(hexToRgb));
      lastQuantization = buildQuantizationModel();
      renderRepresentativeSwatches();
      applyRepresentativeEditPreview();
    },
    isImageLoaded: () => !!currentImageData,
  };
})();
