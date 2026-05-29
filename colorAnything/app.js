(function () {
  const SHARED_IMAGE_EVENT = 'color-anything:image-selected';
  const ASSETS_DIR = 'assets/';
  const SAMPLE_IMAGES = [
    'Sample17.jpg',
    'Sample18.jpg',
    'Sample19.jpg',
    'Sample20.png',
    'Sample10.center.large.jpg',
    'Sample11.center.large.jpg',
    'Sample1.png',
    'Sample2.png',
    'Sample3.png',
    'Sample4.png',
    'Sample5.png',
    'Sample6.jpg',
    'Sample7.jpg',
    'Sample8.jpg',
    'Sample9.jpg',
    'Sample12.jpg',
    'Sample13.jpg',
    'Sample14.jpg',
    'Sample15.jpg',
    'Sample16.jpg'
  ];
  const SAMPLE_IMAGE_CROP_PRESETS = {
    Sample10: { x: 0, y: 0, width: 2000, height: 2000, mode: 'top-left 2000 x 2000 crop' },
    Sample11: { x: 0, y: 0, width: 2000, height: 2000, mode: 'top-left 2000 x 2000 crop' }
  };

  const sharedState = window.ColorAnythingShared || (window.ColorAnythingShared = {});
  let activeSharedObjectUrl = null;

  function toSampleThumbPath(src) {
    return `${ASSETS_DIR}thumbs/${src.replace(/\.[^.]+$/, '.thumb.jpg')}`;
  }

  function getSampleKey(label = '') {
    const match = String(label).match(/(Sample\d+)/i);
    return match ? `${match[1].charAt(0).toUpperCase()}${match[1].slice(1)}` : '';
  }

  function getSampleImageCropPreset(label = '') {
    const sampleKey = getSampleKey(label);
    return sampleKey ? SAMPLE_IMAGE_CROP_PRESETS[sampleKey] || null : null;
  }

  function resolveImageRegion(image, label = '') {
    const preset = getSampleImageCropPreset(label);
    const naturalWidth = image.naturalWidth;
    const naturalHeight = image.naturalHeight;

    if (!preset) {
      return {
        sourceX: 0,
        sourceY: 0,
        sourceWidth: naturalWidth,
        sourceHeight: naturalHeight,
        width: naturalWidth,
        height: naturalHeight,
        mode: 'original size'
      };
    }

    const sourceX = Math.max(0, Math.min(naturalWidth - 1, preset.x));
    const sourceY = Math.max(0, Math.min(naturalHeight - 1, preset.y));
    const sourceWidth = Math.max(1, Math.min(preset.width, naturalWidth - sourceX));
    const sourceHeight = Math.max(1, Math.min(preset.height, naturalHeight - sourceY));

    return {
      sourceX,
      sourceY,
      sourceWidth,
      sourceHeight,
      width: sourceWidth,
      height: sourceHeight,
      mode: preset.mode
    };
  }

  function publishSharedImageSelection(src, label = src, objectUrl = false) {
    if (activeSharedObjectUrl && activeSharedObjectUrl !== src) {
      URL.revokeObjectURL(activeSharedObjectUrl);
      activeSharedObjectUrl = null;
    }

    if (objectUrl) {
      activeSharedObjectUrl = src;
    }

    sharedState.currentSelection = { src, label };
    window.dispatchEvent(new CustomEvent(SHARED_IMAGE_EVENT, {
      detail: sharedState.currentSelection
    }));
  }

  function activateImage(url, label = url, objectUrl = false) {
    publishSharedImageSelection(url, label, objectUrl);
  }

  function setActiveImagePickerNode(activeNode) {
    document.querySelectorAll('.thumb-btn, .upload-btn').forEach(node => {
      node.classList.toggle('active', node === activeNode);
    });
  }

  function buildImageSelector() {
    const container = document.getElementById('imageSelector');
    const uploadInput = document.getElementById('uploadInput');

    if (!container || !uploadInput) {
      return;
    }

    for (const src of SAMPLE_IMAGES) {
      const button = document.createElement('button');
      button.className = 'thumb-btn';
      button.dataset.src = src;
      button.type = 'button';

      const image = document.createElement('img');
      image.src = toSampleThumbPath(src);
      image.alt = src;
      image.loading = 'lazy';
      image.decoding = 'async';
      image.width = 80;
      image.height = 60;
      image.addEventListener('error', () => {
        if (image.src.endsWith(src)) {
          return;
        }
        image.src = src;
      }, { once: true });
      button.appendChild(image);

      button.addEventListener('click', () => {
        setActiveImagePickerNode(button);
        activateImage(ASSETS_DIR + src, src, false);
      });

      container.appendChild(button);
    }

    const uploadButton = document.createElement('button');
    uploadButton.className = 'upload-btn';
    uploadButton.type = 'button';
    uploadButton.innerHTML =
      `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
         <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
         <polyline points="17 8 12 3 7 8"/>
         <line x1="12" y1="3" x2="12" y2="15"/>
       </svg>
       <span>Upload</span>`;
    uploadButton.addEventListener('click', () => uploadInput.click());
    container.appendChild(uploadButton);

    uploadInput.addEventListener('change', event => {
      const file = event.target.files[0];
      if (!file) {
        return;
      }

      const url = URL.createObjectURL(file);
      setActiveImagePickerNode(uploadButton);
      activateImage(url, file.name, true);
      uploadInput.value = '';
    });
  }

  sharedState.sampleImages = SAMPLE_IMAGES;
  sharedState.toSampleThumbPath = toSampleThumbPath;
  sharedState.resolveImageRegion = resolveImageRegion;

  buildImageSelector();

  const firstThumb = document.querySelector('.thumb-btn');
  if (firstThumb) {
    firstThumb.click();
  }
})();
