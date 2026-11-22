/**
 * frontend/js/app.js
 * Primary Application Controller for LeafScan — Plant Disease Detection V2
 * Pure Vanilla JavaScript ES Module (Zero External Dependencies)
 *
 * Implements full state machine:
 *   IDLE -> IMAGE SELECTED / PREVIEW -> PROCESSING -> RESULTS / ERROR
 */

import { diagnoseImage, checkHealth } from './api.js';

// Configuration & Storage Constants
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB (Defensively matched with backend)
const ALLOWED_MIME_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const ALLOWED_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp']);
const STORAGE_KEY = 'leafscan_v2_history_v1';
const MAX_HISTORY_RECORDS = 25;
const MAX_RECENT_SCANS_DISPLAY = 6;
const HEALTH_POLL_INTERVAL_MS = 30000; // 30 seconds

// Scientific Botanical Mapping
const SPECIES_SCIENTIFIC_MAP = {
  'Potato': 'Solanum tuberosum',
  'Tomato': 'Solanum lycopersicum'
};

// Application State
const state = {
  currentView: 'scan',
  selectedFile: null,
  previewObjectUrl: null,
  thumbnailDataUrl: null,
  imageDimensions: { width: null, height: null },
  isDiagnosing: false,
  lastResult: null,
  history: [],
  isBackendHealthy: true,
  healthTimerId: null,
  analyzingStepTimerId: null
};

// DOM Element References
const elements = {
  // Navigation & Shell
  navBtnScan: document.getElementById('nav-btn-scan'),
  navBtnHistory: document.getElementById('nav-btn-history'),
  navBtnAbout: document.getElementById('nav-btn-about'),
  navBrandLogo: document.getElementById('nav-brand-logo'),
  ariaAnnouncer: document.getElementById('aria-status-announcer'),

  // Views
  viewScan: document.getElementById('view-scan'),
  viewAnalyzing: document.getElementById('view-analyzing'),
  viewResult: document.getElementById('view-result'),
  viewHistory: document.getElementById('view-history'),
  viewAbout: document.getElementById('view-about'),

  // In-Page Alert
  uploadAlertBox: document.getElementById('upload-alert-box'),
  uploadAlertTitle: document.getElementById('upload-alert-title'),
  uploadAlertMessage: document.getElementById('upload-alert-message'),
  btnDismissAlert: document.getElementById('btn-dismiss-alert'),

  // State 1: Upload Dropzone & Controls
  uploadZone: document.getElementById('upload-zone'),
  leafInput: document.getElementById('leaf-input'),
  cameraInput: document.getElementById('camera-input'),
  btnBrowseFiles: document.getElementById('btn-browse-files'),
  btnTakePhoto: document.getElementById('btn-take-photo'),

  // State 2: Preview Panel
  previewPanel: document.getElementById('preview-panel'),
  previewImg: document.getElementById('preview-img'),
  previewName: document.getElementById('preview-name'),
  previewSize: document.getElementById('preview-size'),
  previewDimensions: document.getElementById('preview-dimensions'),
  runBtn: document.getElementById('run-btn'),
  changeBtn: document.getElementById('change-btn'),
  retakeBtn: document.getElementById('retake-btn'),
  clearBtn: document.getElementById('clear-btn'),

  // State 3: Workspace Analyzing Card
  workspaceAnalyzingCard: document.getElementById('workspace-analyzing-card'),
  analyzingStepTitle: document.getElementById('analyzing-step-title'),
  analyzingStepDesc: document.getElementById('analyzing-step-desc'),

  // Protocol Accordion
  protocolAccordion: document.querySelector('.protocol-accordion'),
  protocolItems: document.querySelectorAll('.accordion-item'),

  // Section 3: Recent Scans
  sectionRecentScans: document.getElementById('section-recent-scans'),
  recentScansGrid: document.getElementById('recent-scans-grid'),
  recentScansEmpty: document.getElementById('recent-scans-empty'),
  recentScansCount: document.getElementById('recent-scans-count'),
  btnViewAllHistory: document.getElementById('btn-view-all-history'),
  btnLoadSampleEmpty: document.getElementById('btn-load-sample-empty'),

  // Mobile Sticky Bar
  mobileQuickBar: document.getElementById('mobile-quick-bar'),
  btnQuickBrowse: document.getElementById('btn-quick-browse'),
  btnQuickCamera: document.getElementById('btn-quick-camera'),

  // Result View Elements
  resultTimestamp: document.getElementById('result-timestamp'),
  btnResultNewScan: document.getElementById('btn-result-new-scan'),
  btnResultCopySummary: document.getElementById('btn-result-copy-summary'),
  resultSpecimenImg: document.getElementById('result-specimen-img'),
  resultSpecimenFilename: document.getElementById('result-specimen-filename'),
  resultSpecimenMeta: document.getElementById('result-specimen-meta'),

  // Safety Status Banner
  statusBanner: document.getElementById('safety-status-banner'),
  statusIcon: document.getElementById('status-icon'),
  statusPill: document.getElementById('status-pill'),
  statusHeading: document.getElementById('status-heading'),
  statusMessage: document.getElementById('status-message'),
  statusWarnings: document.getElementById('status-warnings'),

  // Species Card (Tier 1)
  speciesScientificName: document.getElementById('species-scientific-name'),
  speciesPredictionName: document.getElementById('species-prediction-name'),
  speciesConfidenceText: document.getElementById('species-confidence-text'),
  gate1StatusPill: document.getElementById('gate1-status-pill'),
  meterValPotato: document.getElementById('meter-val-potato'),
  meterFillPotato: document.getElementById('meter-fill-potato'),
  meterTrackPotato: document.getElementById('meter-track-potato'),
  meterValTomato: document.getElementById('meter-val-tomato'),
  meterFillTomato: document.getElementById('meter-fill-tomato'),
  meterTrackTomato: document.getElementById('meter-track-tomato'),

  // Disease Card (Tier 2)
  gate2StatusPill: document.getElementById('gate2-status-pill'),
  diseasePredictionContent: document.getElementById('disease-prediction-content'),
  diseaseAbstainedContent: document.getElementById('disease-abstained-content'),
  diseasePredictionName: document.getElementById('disease-prediction-name'),
  diseaseConfidenceText: document.getElementById('disease-confidence-text'),
  meterValEarlyBlight: document.getElementById('meter-val-early-blight'),
  meterFillEarlyBlight: document.getElementById('meter-fill-early-blight'),
  meterTrackEarlyBlight: document.getElementById('meter-track-early-blight'),
  meterValLateBlight: document.getElementById('meter-val-late-blight'),
  meterFillLateBlight: document.getElementById('meter-fill-late-blight'),
  meterTrackLateBlight: document.getElementById('meter-track-late-blight'),
  meterValHealthy: document.getElementById('meter-val-healthy'),
  meterFillHealthy: document.getElementById('meter-fill-healthy'),
  meterTrackHealthy: document.getElementById('meter-track-healthy'),

  // Advisory Card
  advisoryTextBox: document.getElementById('advisory-text-box'),

  // Technical Details
  techValVersion: document.getElementById('tech-val-version'),
  techValArchitecture: document.getElementById('tech-val-architecture'),
  techValLatency: document.getElementById('tech-val-latency'),
  techValSafety: document.getElementById('tech-val-safety'),
  techRawJson: document.getElementById('tech-raw-json'),
  btnCopyJson: document.getElementById('btn-copy-json'),

  // History View
  historyCount: document.getElementById('history-count'),
  btnClearHistory: document.getElementById('btn-clear-history'),
  historyEmpty: document.getElementById('history-empty'),
  historyItemsContainer: document.getElementById('history-items-container'),
  btnHistoryStartScan: document.getElementById('btn-history-start-scan')
};

// ============================================================================
// 1. VIEW ROUTING & SCREEN READER ANNOUNCEMENTS
// ============================================================================

/**
 * Switch the active view panel with smooth scrolling and ARIA announcements.
 * @param {'scan'|'analyzing'|'result'|'history'|'about'} viewName
 */
function switchView(viewName) {
  state.currentView = viewName;

  // Toggle view panels
  if (elements.viewScan) elements.viewScan.hidden = (viewName !== 'scan');
  if (elements.viewAnalyzing) elements.viewAnalyzing.hidden = (viewName !== 'analyzing');
  if (elements.viewResult) elements.viewResult.hidden = (viewName !== 'result');
  if (elements.viewHistory) elements.viewHistory.hidden = (viewName !== 'history');
  if (elements.viewAbout) elements.viewAbout.hidden = (viewName !== 'about');

  // Update navigation pills
  const navButtons = [
    { name: 'scan', el: elements.navBtnScan },
    { name: 'history', el: elements.navBtnHistory },
    { name: 'about', el: elements.navBtnAbout }
  ];

  navButtons.forEach(({ name, el }) => {
    if (el) {
      if (name === viewName) {
        el.classList.add('active');
        el.setAttribute('aria-current', 'page');
      } else {
        el.classList.remove('active');
        el.removeAttribute('aria-current');
      }
    }
  });

  // Mobile sticky quick bar is only shown on scan view when not diagnosing
  if (elements.mobileQuickBar) {
    elements.mobileQuickBar.style.display = (viewName === 'scan' && !state.isDiagnosing && !state.selectedFile)
      ? 'flex'
      : 'none';
  }

  // Announce view transition to screen readers
  if (elements.ariaAnnouncer) {
    const labels = {
      scan: 'Foliar diagnosis scan workspace opened.',
      analyzing: 'Analyzing leaf specimen. Please wait.',
      result: 'Diagnostic pathology report loaded.',
      history: 'Diagnostic history log opened.',
      about: 'About technical specifications opened.'
    };
    elements.ariaAnnouncer.textContent = labels[viewName] || `Switched to ${viewName} view.`;
  }

  // Refresh history view when entered
  if (viewName === 'history') {
    renderHistoryView();
  }

  // Refresh recent scans strip when returning to scan view
  if (viewName === 'scan') {
    renderRecentScans();
  }

  // Smooth scroll to top of view
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================================
// 2. BACKEND HEALTH CONNECTION MONITOR
// ============================================================================

/**
 * Probes the health check endpoint asynchronously to validate backend connectivity.
 */
async function pollBackendHealth() {
  try {
    const health = await checkHealth();
    state.isBackendHealthy = Boolean(health && health.status === 'healthy');
  } catch (err) {
    state.isBackendHealthy = false;
  }
}

// ============================================================================
// 3. FILE VALIDATION, UPLOAD & PREVIEW (STATE MACHINE)
// ============================================================================

/**
 * Displays an in-page non-blocking alert banner.
 * @param {string} title
 * @param {string} message
 */
function showUploadAlert(title, message) {
  if (!elements.uploadAlertBox) return;

  if (elements.uploadAlertTitle) elements.uploadAlertTitle.textContent = title;
  if (elements.uploadAlertMessage) elements.uploadAlertMessage.textContent = message;
  elements.uploadAlertBox.hidden = false;

  if (elements.ariaAnnouncer) {
    elements.ariaAnnouncer.textContent = `${title}: ${message}`;
  }
}

/**
 * Hides the in-page alert banner.
 */
function hideUploadAlert() {
  if (elements.uploadAlertBox) {
    elements.uploadAlertBox.hidden = true;
  }
}

/**
 * Validates selected file constraints (format and 10MB size limit).
 * @param {File} file
 * @returns {string|null} Error message or null if valid
 */
function validateFile(file) {
  if (!file) return 'No file was selected.';

  // Check size limit (10 Megabytes)
  if (file.size > MAX_FILE_SIZE_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    return `File size (${sizeMb} MB) exceeds maximum allowed limit of 10 MB.`;
  }

  // Check MIME type and extension
  const mime = (file.type || '').toLowerCase();
  const dotIndex = file.name.lastIndexOf('.');
  const ext = dotIndex !== -1 ? file.name.slice(dotIndex).toLowerCase() : '';

  const hasValidMime = ALLOWED_MIME_TYPES.has(mime);
  const hasValidExt = ALLOWED_EXTENSIONS.has(ext);

  if (!hasValidMime && !hasValidExt) {
    return `Unsupported image format (${file.type || ext || 'unknown'}). Supported formats: JPEG, PNG, WebP.`;
  }

  return null;
}

/**
 * Generates a lightweight, compact thumbnail data URL for offline storage.
 * @param {HTMLImageElement} img
 * @returns {string} base64 data URL
 */
function generateThumbnailDataUrl(img) {
  try {
    const canvas = document.createElement('canvas');
    const maxSize = 120;
    let w = img.naturalWidth || img.width || 120;
    let h = img.naturalHeight || img.height || 120;

    if (w > h) {
      if (w > maxSize) {
        h = Math.round((h * maxSize) / w);
        w = maxSize;
      }
    } else {
      if (h > maxSize) {
        w = Math.round((w * maxSize) / h);
        h = maxSize;
      }
    }

    canvas.width = Math.max(w, 1);
    canvas.height = Math.max(h, 1);

    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.7);
  } catch (err) {
    console.warn('Could not generate thumbnail data URL:', err);
    return null;
  }
}

/**
 * Selects and processes an image file, updating preview and dimensions.
 * @param {File} file
 */
function selectFile(file) {
  hideUploadAlert();

  const validationError = validateFile(file);
  if (validationError) {
    showUploadAlert('Invalid Image File', validationError);
    return;
  }

  // Revoke prior Object URL to prevent browser memory leaks
  if (state.previewObjectUrl) {
    URL.revokeObjectURL(state.previewObjectUrl);
    state.previewObjectUrl = null;
  }

  state.selectedFile = file;
  state.previewObjectUrl = URL.createObjectURL(file);
  state.thumbnailDataUrl = null;
  state.imageDimensions = { width: null, height: null };

  // Calculate image natural dimensions asynchronously
  const tempImg = new Image();
  tempImg.onload = () => {
    state.imageDimensions.width = tempImg.naturalWidth;
    state.imageDimensions.height = tempImg.naturalHeight;
    if (elements.previewDimensions) {
      elements.previewDimensions.textContent = `${tempImg.naturalWidth} × ${tempImg.naturalHeight} px`;
    }
    // Generate compact thumbnail for history
    state.thumbnailDataUrl = generateThumbnailDataUrl(tempImg);
  };
  tempImg.src = state.previewObjectUrl;

  // Update Preview UI Elements
  if (elements.previewImg) {
    elements.previewImg.src = state.previewObjectUrl;
    elements.previewImg.alt = `Preview of selected leaf specimen ${file.name}`;
  }
  if (elements.previewName) {
    elements.previewName.textContent = file.name;
    elements.previewName.title = file.name;
  }
  if (elements.previewSize) {
    elements.previewSize.textContent = formatBytes(file.size);
  }
  if (elements.previewDimensions) {
    elements.previewDimensions.textContent = 'Calculating...';
  }

  // Transition UI: Hide upload zone, show preview panel
  if (elements.uploadZone) elements.uploadZone.hidden = true;
  if (elements.previewPanel) elements.previewPanel.hidden = false;
  if (elements.workspaceAnalyzingCard) elements.workspaceAnalyzingCard.hidden = true;
  if (elements.runBtn) elements.runBtn.disabled = false;

  // Update mobile quick bar
  if (elements.mobileQuickBar) elements.mobileQuickBar.style.display = 'none';

  if (elements.ariaAnnouncer) {
    elements.ariaAnnouncer.textContent = `Selected leaf photo ${file.name} ready for diagnosis.`;
  }
}

/**
 * Resets the upload interface back to the Idle State.
 */
function resetUpload() {
  hideUploadAlert();

  if (state.previewObjectUrl) {
    URL.revokeObjectURL(state.previewObjectUrl);
    state.previewObjectUrl = null;
  }
  state.selectedFile = null;
  state.thumbnailDataUrl = null;
  state.imageDimensions = { width: null, height: null };

  if (elements.leafInput) elements.leafInput.value = '';
  if (elements.cameraInput) elements.cameraInput.value = '';

  if (elements.previewImg) elements.previewImg.src = '';
  if (elements.previewName) {
    elements.previewName.textContent = 'Selected image';
    elements.previewName.removeAttribute('title');
  }
  if (elements.previewSize) elements.previewSize.textContent = '0 KB';
  if (elements.previewDimensions) elements.previewDimensions.textContent = '-- × -- px';

  if (elements.previewPanel) elements.previewPanel.hidden = true;
  if (elements.workspaceAnalyzingCard) elements.workspaceAnalyzingCard.hidden = true;
  if (elements.uploadZone) elements.uploadZone.hidden = false;
  if (elements.runBtn) elements.runBtn.disabled = true;

  if (state.currentView === 'scan' && elements.mobileQuickBar) {
    elements.mobileQuickBar.style.display = 'flex';
  }

  if (elements.ariaAnnouncer) {
    elements.ariaAnnouncer.textContent = 'Upload area reset. Ready for new leaf image.';
  }
}

/**
 * Loads a verified genuine sample leaf photo from the project repository.
 */
async function loadSampleLeaf() {
  try {
    const res = await fetch('assets/sample_leaf.jpg');
    if (!res.ok) {
      throw new Error(`Failed to load sample image asset (HTTP ${res.status})`);
    }
    const blob = await res.blob();
    const sampleFile = new File([blob], 'potato_early_blight_sample.jpg', { type: 'image/jpeg' });
    selectFile(sampleFile);
  } catch (err) {
    console.warn('Sample leaf load error:', err);
    showUploadAlert('Sample Specimen Unavailable', 'Could not load the project sample leaf. Please browse a photo from your device.');
  }
}

// ============================================================================
// 4. DIAGNOSIS WORKFLOW & HONEST INFERENCE PROGRESS
// ============================================================================

const INFERENCE_STEPS = [
  {
    title: 'Preparing Foliar Specimen...',
    desc: 'Verifying image resolution, normalization, and RGB raster integrity.'
  },
  {
    title: 'Extracting Visual Features...',
    desc: 'Executing MobileNetV2 botanist backbone to assess plant morphology.'
  },
  {
    title: 'Evaluating Botanical Identity (Gate 1)...',
    desc: 'Assessing Solanum tuberosum vs Solanum lycopersicum confidence threshold.'
  },
  {
    title: 'Processing CBAM Attention Submodules...',
    desc: 'Refining channel and spatial attention weights for foliar necrotic lesions.'
  },
  {
    title: 'Synthesizing Diagnostic Pathology Report...',
    desc: 'Evaluating Late Blight escalation boundaries and agronomic decision support.'
  }
];

/**
 * Starts rotating indeterminate progress messages during model inference.
 */
function startInferenceProgressIndicator() {
  let stepIndex = 0;

  function updateStep() {
    const step = INFERENCE_STEPS[stepIndex % INFERENCE_STEPS.length];
    if (elements.analyzingStepTitle) elements.analyzingStepTitle.textContent = step.title;
    if (elements.analyzingStepDesc) elements.analyzingStepDesc.textContent = step.desc;
    stepIndex++;
  }

  updateStep();
  state.analyzingStepTimerId = setInterval(updateStep, 1800);
}

/**
 * Stops the progress step rotation.
 */
function stopInferenceProgressIndicator() {
  if (state.analyzingStepTimerId) {
    clearInterval(state.analyzingStepTimerId);
    state.analyzingStepTimerId = null;
  }
}

/**
 * Executes the complete diagnostic workflow.
 */
async function executeDiagnosis() {
  if (!state.selectedFile || state.isDiagnosing) return;

  hideUploadAlert();
  state.isDiagnosing = true;

  // Prevent duplicate submissions
  if (elements.runBtn) elements.runBtn.disabled = true;
  if (elements.changeBtn) elements.changeBtn.disabled = true;
  if (elements.retakeBtn) elements.retakeBtn.disabled = true;
  if (elements.clearBtn) elements.clearBtn.disabled = true;

  // Transition UI to State 3: Workspace Analyzing Card
  if (elements.previewPanel) elements.previewPanel.hidden = true;
  if (elements.uploadZone) elements.uploadZone.hidden = true;
  if (elements.workspaceAnalyzingCard) elements.workspaceAnalyzingCard.hidden = false;

  startInferenceProgressIndicator();

  if (elements.ariaAnnouncer) {
    elements.ariaAnnouncer.textContent = 'Diagnostic inference started. Running MobileNetV2 and CBAM screening.';
  }

  try {
    const result = await diagnoseImage(state.selectedFile);

    stopInferenceProgressIndicator();
    state.lastResult = result;
    state.isDiagnosing = false;

    // Persist diagnosis to localStorage history (including thumbnail)
    saveToHistory(state.selectedFile.name, result, state.thumbnailDataUrl);

    // Render result dashboard
    renderDiagnosisResult(result);

    // Transition to Results View
    switchView('result');

  } catch (error) {
    stopInferenceProgressIndicator();
    state.isDiagnosing = false;

    if (elements.runBtn) elements.runBtn.disabled = false;
    if (elements.changeBtn) elements.changeBtn.disabled = false;
    if (elements.retakeBtn) elements.retakeBtn.disabled = false;
    if (elements.clearBtn) elements.clearBtn.disabled = false;

    // Restore preview panel in workspace so user can retry or change photo
    if (elements.workspaceAnalyzingCard) elements.workspaceAnalyzingCard.hidden = true;
    if (elements.previewPanel) elements.previewPanel.hidden = false;

    console.error('Diagnosis failure:', error);

    // Display non-blocking alert in workspace
    showUploadAlert('Diagnosis Inference Failed', error.message || 'Server error occurred during classification. Please retry.');
  }
}

// ============================================================================
// 5. DIAGNOSIS RESULTS RENDERING
// ============================================================================

/**
 * Renders the structured prediction response in the results view.
 * @param {Object} res - JSON prediction payload
 */
function renderDiagnosisResult(res) {
  if (elements.resultTimestamp) {
    elements.resultTimestamp.textContent = `Evaluated on ${new Date().toLocaleString()}`;
  }

  // 1. Render Specimen Image and Meta in Left Column
  if (elements.resultSpecimenImg && state.previewObjectUrl) {
    elements.resultSpecimenImg.src = state.previewObjectUrl;
    elements.resultSpecimenImg.alt = `Analyzed specimen: ${res.diagnosis || 'Foliar Leaf'}`;
  }
  if (elements.resultSpecimenFilename) {
    elements.resultSpecimenFilename.textContent = state.selectedFile?.name || 'leaf_specimen.jpg';
  }
  if (elements.resultSpecimenMeta) {
    const dimText = (state.imageDimensions.width && state.imageDimensions.height)
      ? `${state.imageDimensions.width} × ${state.imageDimensions.height} px`
      : '224 × 224 RGB';
    elements.resultSpecimenMeta.textContent = dimText;
  }

  // 2. Safety Status Banner Mapping (4 explicit backend statuses)
  renderSafetyBanner(res.safety_status, res.safety_warnings, res.plant_species, res.disease);

  // 3. Botanical Species Identification Card (Tier 1)
  const species = res.plant_species || 'Unknown';
  if (elements.speciesPredictionName) elements.speciesPredictionName.textContent = species;
  if (elements.speciesScientificName) elements.speciesScientificName.textContent = SPECIES_SCIENTIFIC_MAP[species] || 'Solanaceae family';

  const speciesConf = typeof res.species_confidence === 'number' ? (res.species_confidence * 100) : 0;
  if (elements.speciesConfidenceText) elements.speciesConfidenceText.textContent = `${speciesConf.toFixed(2)}%`;

  // Gate 1 status pill
  if (elements.gate1StatusPill) {
    if (res.safety_status === 'species_uncertain') {
      elements.gate1StatusPill.textContent = 'Gate 1 Triggered (<90%)';
      elements.gate1StatusPill.style.color = 'var(--status-species-uncertain)';
      elements.gate1StatusPill.style.borderColor = 'var(--status-species-uncertain-border)';
    } else {
      elements.gate1StatusPill.textContent = 'Gate 1 Passed (≥90%)';
      elements.gate1StatusPill.style.color = 'var(--status-confident)';
      elements.gate1StatusPill.style.borderColor = 'var(--status-confident-border)';
    }
  }

  // Species distribution meters
  const speciesProbs = res.species_probabilities || {};
  const potatoProb = (speciesProbs['Potato'] || 0) * 100;
  const tomatoProb = (speciesProbs['Tomato'] || 0) * 100;

  if (elements.meterValPotato) elements.meterValPotato.textContent = `${potatoProb.toFixed(1)}%`;
  if (elements.meterFillPotato) elements.meterFillPotato.style.width = `${Math.min(potatoProb, 100)}%`;
  if (elements.meterTrackPotato) elements.meterTrackPotato.setAttribute('aria-valuenow', potatoProb.toFixed(1));

  if (elements.meterValTomato) elements.meterValTomato.textContent = `${tomatoProb.toFixed(1)}%`;
  if (elements.meterFillTomato) elements.meterFillTomato.style.width = `${Math.min(tomatoProb, 100)}%`;
  if (elements.meterTrackTomato) elements.meterTrackTomato.setAttribute('aria-valuenow', tomatoProb.toFixed(1));

  // 4. Foliar Pathology Specialist Card (Tier 2)
  const disease = res.disease;
  if (disease) {
    // Normal diagnosis state
    if (elements.diseasePredictionContent) elements.diseasePredictionContent.hidden = false;
    if (elements.diseaseAbstainedContent) elements.diseaseAbstainedContent.hidden = true;

    if (elements.diseasePredictionName) elements.diseasePredictionName.textContent = disease;
    const diseaseConf = typeof res.disease_confidence === 'number' ? (res.disease_confidence * 100) : 0;
    if (elements.diseaseConfidenceText) elements.diseaseConfidenceText.textContent = `${diseaseConf.toFixed(2)}%`;

    // Gate 2 pill text
    if (elements.gate2StatusPill) {
      if (res.safety_status === 'late_blight_uncertain') {
        elements.gate2StatusPill.textContent = 'Late Blight Warning (<90%)';
        elements.gate2StatusPill.style.color = 'var(--status-late-blight)';
        elements.gate2StatusPill.style.borderColor = 'var(--status-late-blight-border)';
      } else if (res.safety_status === 'disease_uncertain') {
        elements.gate2StatusPill.textContent = 'Marginal Confidence (<70%)';
        elements.gate2StatusPill.style.color = 'var(--status-disease-uncertain)';
        elements.gate2StatusPill.style.borderColor = 'var(--status-disease-uncertain-border)';
      } else {
        elements.gate2StatusPill.textContent = 'Specialist Confident';
        elements.gate2StatusPill.style.color = 'var(--status-confident)';
        elements.gate2StatusPill.style.borderColor = 'var(--status-confident-border)';
      }
    }

    // Disease distribution meters
    const diseaseProbs = res.probabilities || {};
    const ebProb = (diseaseProbs['Early Blight'] || 0) * 100;
    const lbProb = (diseaseProbs['Late Blight'] || 0) * 100;
    const healthyProb = (diseaseProbs['Healthy'] || 0) * 100;

    if (elements.meterValEarlyBlight) elements.meterValEarlyBlight.textContent = `${ebProb.toFixed(1)}%`;
    if (elements.meterFillEarlyBlight) elements.meterFillEarlyBlight.style.width = `${Math.min(ebProb, 100)}%`;
    if (elements.meterTrackEarlyBlight) elements.meterTrackEarlyBlight.setAttribute('aria-valuenow', ebProb.toFixed(1));

    if (elements.meterValLateBlight) elements.meterValLateBlight.textContent = `${lbProb.toFixed(1)}%`;
    if (elements.meterFillLateBlight) elements.meterFillLateBlight.style.width = `${Math.min(lbProb, 100)}%`;
    if (elements.meterTrackLateBlight) elements.meterTrackLateBlight.setAttribute('aria-valuenow', lbProb.toFixed(1));

    if (elements.meterValHealthy) elements.meterValHealthy.textContent = `${healthyProb.toFixed(1)}%`;
    if (elements.meterFillHealthy) elements.meterFillHealthy.style.width = `${Math.min(healthyProb, 100)}%`;
    if (elements.meterTrackHealthy) elements.meterTrackHealthy.setAttribute('aria-valuenow', healthyProb.toFixed(1));

  } else {
    // Abstained state (Species uncertain)
    if (elements.diseasePredictionContent) elements.diseasePredictionContent.hidden = true;
    if (elements.diseaseAbstainedContent) elements.diseaseAbstainedContent.hidden = false;
    if (elements.gate2StatusPill) {
      elements.gate2StatusPill.textContent = 'Abstained by Gate 1';
      elements.gate2StatusPill.style.color = 'var(--status-species-uncertain)';
      elements.gate2StatusPill.style.borderColor = 'var(--status-species-uncertain-border)';
    }
  }

  // 5. Actionable Agronomic Advisory Card
  if (elements.advisoryTextBox) {
    elements.advisoryTextBox.textContent = res.advisory || 'No specific advisory returned.';
    elements.advisoryTextBox.classList.remove('urgent', 'warning');

    if (res.safety_status === 'late_blight_uncertain') {
      elements.advisoryTextBox.classList.add('urgent');
    } else if (res.safety_status === 'disease_uncertain' || res.safety_status === 'species_uncertain') {
      elements.advisoryTextBox.classList.add('warning');
    }
  }

  // 6. Technical Metadata
  if (elements.techValVersion) elements.techValVersion.textContent = res.model_version || 'v2';
  if (elements.techValArchitecture) elements.techValArchitecture.textContent = res.metadata?.architecture || 'MobileNetV2 + CBAM';
  if (elements.techValLatency) {
    elements.techValLatency.textContent = res.metadata?.execution_time_ms != null
      ? `${res.metadata.execution_time_ms.toFixed(1)} ms`
      : '-- ms';
  }
  if (elements.techValSafety) elements.techValSafety.textContent = res.safety_status || 'unknown';
  if (elements.techRawJson) elements.techRawJson.textContent = JSON.stringify(res, null, 2);
}

/**
 * Maps backend safety_status to status banner styling and messaging.
 * @param {string} status
 * @param {Array<string>} warnings
 * @param {string} plantSpecies
 * @param {string|null} disease
 */
function renderSafetyBanner(status, warnings, plantSpecies, disease) {
  const banner = elements.statusBanner;
  if (!banner) return;

  banner.className = 'status-banner'; // Reset class list

  if (elements.statusWarnings) {
    elements.statusWarnings.innerHTML = '';
    elements.statusWarnings.hidden = true;
  }

  switch (status) {
    case 'confident':
      banner.classList.add('confident');
      if (elements.statusIcon) elements.statusIcon.textContent = '✓';
      if (elements.statusPill) elements.statusPill.textContent = 'Confident Diagnosis';
      if (elements.statusHeading) elements.statusHeading.textContent = `${plantSpecies} — ${disease || 'Foliar Screening'}`;
      if (elements.statusMessage) elements.statusMessage.textContent = 'Prediction satisfies all safety gates. Diagnostic confidence is optimal.';
      break;

    case 'species_uncertain':
      banner.classList.add('species_uncertain');
      if (elements.statusIcon) elements.statusIcon.textContent = '⚠️';
      if (elements.statusPill) elements.statusPill.textContent = 'Species Identification Uncertain';
      if (elements.statusHeading) elements.statusHeading.textContent = 'Safety Gate 1 Triggered (<90% Species Confidence)';
      if (elements.statusMessage) elements.statusMessage.textContent = 'Botanical species identification is marginal. Disease classification has been abstained to avoid crop misdiagnosis. Do NOT apply chemicals to unverified plant species.';
      break;

    case 'late_blight_uncertain':
      banner.classList.add('late_blight_uncertain');
      if (elements.statusIcon) elements.statusIcon.textContent = '🛑';
      if (elements.statusPill) elements.statusPill.textContent = 'High Consequence Risk';
      if (elements.statusHeading) elements.statusHeading.textContent = 'Safety Gate 2: Late Blight Uncertainty (<90%)';
      if (elements.statusMessage) elements.statusMessage.textContent = 'Potential foliar lesions detected, but Late Blight confidence is below the safety threshold. DO NOT apply aggressive systemic fungicides at this stage. Isolate specimen and re-inspect within 24 to 48 hours.';
      break;

    case 'disease_uncertain':
      banner.classList.add('disease_uncertain');
      if (elements.statusIcon) elements.statusIcon.textContent = '⚡';
      if (elements.statusPill) elements.statusPill.textContent = 'Marginal Confidence';
      if (elements.statusHeading) elements.statusHeading.textContent = 'Diagnostic Confidence Below Threshold (<70%)';
      if (elements.statusMessage) elements.statusMessage.textContent = 'Visual foliar symptoms are ambiguous. Re-photograph the leaf in diffuse daylight and monitor closely before initiating chemical treatment.';
      break;

    default:
      banner.classList.add('species_uncertain');
      if (elements.statusIcon) elements.statusIcon.textContent = 'ℹ️';
      if (elements.statusPill) elements.statusPill.textContent = 'Notice';
      if (elements.statusHeading) elements.statusHeading.textContent = 'Safety Assessment Completed';
      if (elements.statusMessage) elements.statusMessage.textContent = 'Result received with status: ' + (status || 'unspecified');
      break;
  }

  // Populate warning list if present
  if (Array.isArray(warnings) && warnings.length > 0 && elements.statusWarnings) {
    warnings.forEach(warningText => {
      const li = document.createElement('li');
      li.textContent = warningText;
      elements.statusWarnings.appendChild(li);
    });
    elements.statusWarnings.hidden = false;
  }
}

/**
 * Copies a structured text diagnostic summary to the clipboard.
 */
async function copyDiagnosticSummary() {
  if (!state.lastResult) return;

  const res = state.lastResult;
  const timeStr = new Date().toLocaleString();
  const summaryText = `[LeafScan V2 Diagnostic Pathology Report]
Evaluated: ${timeStr}
Specimen: ${state.selectedFile?.name || 'leaf_specimen.jpg'}
Species: ${res.plant_species || 'Unknown'} (${((res.species_confidence || 0) * 100).toFixed(1)}% confidence)
Diagnosis: ${res.diagnosis || 'Abstained'}
Specialist Confidence: ${res.disease_confidence != null ? ((res.disease_confidence * 100).toFixed(1) + '%') : 'N/A'}
Safety Decision: ${res.safety_status}
Advisory: ${res.advisory || 'None'}
Architecture: MobileNetV2 + CBAM Attention (Opset 17)`;

  try {
    await navigator.clipboard.writeText(summaryText);
    if (elements.btnResultCopySummary) {
      const origHtml = elements.btnResultCopySummary.innerHTML;
      elements.btnResultCopySummary.innerHTML = '<span aria-hidden="true">✓</span> Copied!';
      setTimeout(() => {
        elements.btnResultCopySummary.innerHTML = origHtml;
      }, 2000);
    }
  } catch (err) {
    console.warn('Clipboard write failed:', err);
  }
}

// ============================================================================
// 6. LOCAL HISTORY & SECTION 3 RECENT SCANS
// ============================================================================

/**
 * Loads diagnostic history from localStorage.
 * @returns {Array<Object>}
 */
function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.warn('Failed to load local scan history:', err);
    return [];
  }
}

/**
 * Appends a successful scan record to localStorage.
 * @param {string} filename
 * @param {Object} res
 * @param {string|null} thumbnailDataUrl
 */
function saveToHistory(filename, res, thumbnailDataUrl) {
  try {
    const records = loadHistory();
    const newEntry = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      timestamp: new Date().toISOString(),
      filename: filename || 'leaf_specimen.jpg',
      species: res.plant_species || 'Unknown',
      disease: res.disease || 'Abstained',
      confidence: res.disease_confidence != null ? res.disease_confidence : res.species_confidence,
      safetyStatus: res.safety_status || 'confident',
      diagnosis: res.diagnosis || 'Plant Screening',
      thumbnail: thumbnailDataUrl || null,
      resultData: res
    };

    records.unshift(newEntry);

    if (records.length > MAX_HISTORY_RECORDS) {
      records.length = MAX_HISTORY_RECORDS;
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
    state.history = records;
  } catch (err) {
    console.warn('Failed to persist scan to localStorage:', err);
  }
}

/**
 * Clears all local diagnostic history.
 */
function clearAllHistory() {
  try {
    localStorage.removeItem(STORAGE_KEY);
    state.history = [];
    renderHistoryView();
    renderRecentScans();
    if (elements.ariaAnnouncer) {
      elements.ariaAnnouncer.textContent = 'Local diagnostic history successfully cleared.';
    }
  } catch (err) {
    console.warn('Error clearing localStorage:', err);
  }
}

/**
 * Renders the Recent Scans cards on the Scan View (Section 3).
 */
function renderRecentScans() {
  if (!elements.recentScansGrid || !elements.recentScansEmpty) return;

  const records = loadHistory();
  state.history = records;

  if (elements.recentScansCount) {
    elements.recentScansCount.textContent = `${records.length} scan${records.length === 1 ? '' : 's'} recorded`;
  }

  if (records.length === 0) {
    elements.recentScansEmpty.hidden = false;
    elements.recentScansGrid.innerHTML = '';
    return;
  }

  elements.recentScansEmpty.hidden = true;
  elements.recentScansGrid.innerHTML = '';

  const displayRecords = records.slice(0, MAX_RECENT_SCANS_DISPLAY);

  displayRecords.forEach(rec => {
    const card = document.createElement('div');
    card.className = 'recent-scan-card';
    card.setAttribute('tabindex', '0');
    card.setAttribute('role', 'button');
    card.setAttribute('aria-label', `View diagnosis for ${rec.diagnosis || rec.species}`);

    // Thumbnail
    const thumbWrap = document.createElement('div');
    thumbWrap.className = 'recent-thumbnail-wrap';
    if (rec.thumbnail) {
      const img = document.createElement('img');
      img.src = rec.thumbnail;
      img.alt = rec.filename || 'Leaf thumbnail';
      img.className = 'recent-thumbnail-img';
      thumbWrap.appendChild(img);
    } else {
      const icon = document.createElement('span');
      icon.className = 'recent-thumbnail-fallback';
      icon.textContent = '🍃';
      thumbWrap.appendChild(icon);
    }

    // Info
    const info = document.createElement('div');
    info.className = 'recent-scan-info';

    const title = document.createElement('div');
    title.className = 'recent-scan-title';
    title.textContent = rec.diagnosis || `${rec.species} - ${rec.disease}`;

    const meta = document.createElement('div');
    meta.className = 'recent-scan-meta';
    const confVal = typeof rec.confidence === 'number' ? (rec.confidence * 100).toFixed(1) : '--';
    const dateStr = formatRelativeTime(rec.timestamp);
    meta.textContent = `${dateStr} · ${confVal}% confidence`;

    const footer = document.createElement('div');
    footer.className = 'recent-scan-footer';

    const statusBadge = document.createElement('span');
    statusBadge.className = 'tag-badge';
    statusBadge.textContent = rec.safetyStatus;

    if (rec.safetyStatus === 'confident') {
      statusBadge.style.color = 'var(--status-confident)';
    } else if (rec.safetyStatus === 'late_blight_uncertain') {
      statusBadge.style.color = 'var(--status-late-blight)';
    } else if (rec.safetyStatus === 'species_uncertain') {
      statusBadge.style.color = 'var(--status-species-uncertain)';
    } else {
      statusBadge.style.color = 'var(--status-disease-uncertain)';
    }

    footer.appendChild(statusBadge);

    info.appendChild(title);
    info.appendChild(meta);
    info.appendChild(footer);

    card.appendChild(thumbWrap);
    card.appendChild(info);

    // Click handler to open report
    const openRecord = () => {
      if (rec.resultData) {
        renderDiagnosisResult(rec.resultData);
        switchView('result');
      }
    };

    card.addEventListener('click', openRecord);
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openRecord();
      }
    });

    elements.recentScansGrid.appendChild(card);
  });
}

/**
 * Renders the full history view list.
 */
function renderHistoryView() {
  const records = loadHistory();
  state.history = records;

  if (elements.historyCount) {
    elements.historyCount.textContent = `${records.length} scan${records.length === 1 ? '' : 's'} recorded`;
  }
  if (elements.btnClearHistory) {
    elements.btnClearHistory.disabled = (records.length === 0);
  }

  if (records.length === 0) {
    if (elements.historyEmpty) elements.historyEmpty.hidden = false;
    if (elements.historyItemsContainer) {
      elements.historyItemsContainer.hidden = true;
      elements.historyItemsContainer.innerHTML = '';
    }
    return;
  }

  if (elements.historyEmpty) elements.historyEmpty.hidden = true;
  if (elements.historyItemsContainer) {
    elements.historyItemsContainer.hidden = false;
    elements.historyItemsContainer.innerHTML = '';

    records.forEach(rec => {
      const card = document.createElement('div');
      card.className = 'history-card';

      const leftDiv = document.createElement('div');
      leftDiv.style.display = 'flex';
      leftDiv.style.alignItems = 'center';
      leftDiv.style.gap = '12px';

      if (rec.thumbnail) {
        const thumbWrap = document.createElement('div');
        thumbWrap.className = 'recent-thumbnail-wrap';
        const img = document.createElement('img');
        img.src = rec.thumbnail;
        img.alt = rec.filename || 'Leaf thumbnail';
        img.className = 'recent-thumbnail-img';
        thumbWrap.appendChild(img);
        leftDiv.appendChild(thumbWrap);
      }

      const infoDiv = document.createElement('div');

      const timeSpan = document.createElement('div');
      timeSpan.className = 'history-time';
      const dateObj = new Date(rec.timestamp);
      timeSpan.textContent = isNaN(dateObj) ? rec.timestamp : dateObj.toLocaleString();

      const titleDiv = document.createElement('div');
      titleDiv.className = 'history-title';
      titleDiv.textContent = rec.diagnosis || `${rec.species} - ${rec.disease}`;

      const metaDiv = document.createElement('div');
      metaDiv.style.fontSize = '12px';
      metaDiv.style.color = 'var(--color-muted)';
      const confVal = typeof rec.confidence === 'number' ? (rec.confidence * 100).toFixed(1) : '--';
      metaDiv.textContent = `File: ${rec.filename} · Confidence: ${confVal}%`;

      infoDiv.appendChild(timeSpan);
      infoDiv.appendChild(titleDiv);
      infoDiv.appendChild(metaDiv);
      leftDiv.appendChild(infoDiv);

      const actionsDiv = document.createElement('div');
      actionsDiv.className = 'history-tag-group';

      const statusBadge = document.createElement('span');
      statusBadge.className = 'tag-badge';
      statusBadge.textContent = rec.safetyStatus;

      if (rec.safetyStatus === 'confident') {
        statusBadge.style.color = 'var(--status-confident)';
      } else if (rec.safetyStatus === 'late_blight_uncertain') {
        statusBadge.style.color = 'var(--status-late-blight)';
      } else if (rec.safetyStatus === 'species_uncertain') {
        statusBadge.style.color = 'var(--status-species-uncertain)';
      } else {
        statusBadge.style.color = 'var(--status-disease-uncertain)';
      }

      const btnView = document.createElement('button');
      btnView.type = 'button';
      btnView.className = 'btn btn-secondary btn-sm';
      btnView.textContent = 'View Report';
      btnView.addEventListener('click', () => {
        if (rec.resultData) {
          renderDiagnosisResult(rec.resultData);
          switchView('result');
        }
      });

      actionsDiv.appendChild(statusBadge);
      actionsDiv.appendChild(btnView);

      card.appendChild(leftDiv);
      card.appendChild(actionsDiv);
      elements.historyItemsContainer.appendChild(card);
    });
  }
}

// ============================================================================
// 7. INTERACTIVE ACCORDION & PROTOCOL HANDLERS
// ============================================================================

/**
 * Initializes interactive accordion toggles for the Diagnostic Imaging Protocol.
 */
function setupProtocolAccordion() {
  const items = document.querySelectorAll('.accordion-item');

  items.forEach(item => {
    const headerBtn = item.querySelector('.accordion-header');
    const body = item.querySelector('.accordion-body');
    if (!headerBtn || !body) return;

    headerBtn.addEventListener('click', () => {
      const isExpanded = item.classList.contains('expanded');

      // Accessible toggle
      if (isExpanded) {
        item.classList.remove('expanded');
        headerBtn.setAttribute('aria-expanded', 'false');
        body.hidden = true;
      } else {
        item.classList.add('expanded');
        headerBtn.setAttribute('aria-expanded', 'true');
        body.hidden = false;
      }
    });
  });
}

// ============================================================================
// 8. EVENT LISTENERS & LIFECYCLE
// ============================================================================

function setupEventListeners() {
  // Navigation Tabs
  elements.navBtnScan?.addEventListener('click', () => switchView('scan'));
  elements.navBtnHistory?.addEventListener('click', () => switchView('history'));
  elements.navBtnAbout?.addEventListener('click', () => switchView('about'));
  elements.navBrandLogo?.addEventListener('click', (e) => {
    e.preventDefault();
    switchView('scan');
  });

  // Browse Files CTA Click
  elements.btnBrowseFiles?.addEventListener('click', (e) => {
    e.stopPropagation();
    elements.leafInput?.click();
  });

  // Camera CTA Click
  elements.btnTakePhoto?.addEventListener('click', (e) => {
    e.stopPropagation();
    elements.cameraInput?.click();
  });

  // Mobile Sticky Bar Triggers
  elements.btnQuickBrowse?.addEventListener('click', () => {
    elements.leafInput?.click();
  });

  elements.btnQuickCamera?.addEventListener('click', () => {
    elements.cameraInput?.click();
  });

  // Load Sample in Empty Recent Scans
  elements.btnLoadSampleEmpty?.addEventListener('click', () => {
    loadSampleLeaf();
  });

  // Alert Dismiss
  elements.btnDismissAlert?.addEventListener('click', hideUploadAlert);

  // Upload Zone Container Click (fall-through to file input)
  elements.uploadZone?.addEventListener('click', (e) => {
    // Only trigger if click wasn't on an explicit child button
    if (!e.target.closest('button')) {
      elements.leafInput?.click();
    }
  });

  elements.uploadZone?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      if (!e.target.closest('button')) {
        e.preventDefault();
        elements.leafInput?.click();
      }
    }
  });

  // Standard File Input Change
  elements.leafInput?.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      selectFile(files[0]);
    }
  });

  // Camera File Input Change
  elements.cameraInput?.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      selectFile(files[0]);
    }
  });

  // Drag & Drop Handlers with counter to avoid flickering on child element hover
  let dragCounter = 0;

  elements.uploadZone?.addEventListener('dragenter', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter++;
    elements.uploadZone.classList.add('dragover');
  });

  elements.uploadZone?.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    elements.uploadZone.classList.add('dragover');
  });

  elements.uploadZone?.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter--;
    if (dragCounter <= 0) {
      dragCounter = 0;
      elements.uploadZone.classList.remove('dragover');
    }
  });

  elements.uploadZone?.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter = 0;
    elements.uploadZone.classList.remove('dragover');

    const dt = e.dataTransfer;
    const files = dt?.files;
    if (files && files.length > 0) {
      selectFile(files[0]);
    }
  });

  // Preview Panel Action Buttons
  elements.runBtn?.addEventListener('click', executeDiagnosis);
  elements.changeBtn?.addEventListener('click', () => {
    elements.leafInput?.click();
  });
  elements.retakeBtn?.addEventListener('click', () => {
    elements.cameraInput?.click();
  });
  elements.clearBtn?.addEventListener('click', resetUpload);

  // Result View Actions
  elements.btnResultNewScan?.addEventListener('click', () => {
    resetUpload();
    switchView('scan');
  });

  elements.btnResultCopySummary?.addEventListener('click', copyDiagnosticSummary);

  // Copy Raw JSON Payload
  elements.btnCopyJson?.addEventListener('click', async () => {
    const rawText = elements.techRawJson?.textContent;
    if (!rawText) return;
    try {
      await navigator.clipboard.writeText(rawText);
      const originalText = elements.btnCopyJson.textContent;
      elements.btnCopyJson.textContent = 'Copied!';
      setTimeout(() => {
        elements.btnCopyJson.textContent = originalText;
      }, 2000);
    } catch (err) {
      console.warn('Clipboard write failed:', err);
    }
  });

  // Section 3: View All History Button
  elements.btnViewAllHistory?.addEventListener('click', () => {
    switchView('history');
  });

  // History Tab Actions
  elements.btnClearHistory?.addEventListener('click', () => {
    if (confirm('Clear all local scan history? This action cannot be undone.')) {
      clearAllHistory();
    }
  });

  elements.btnHistoryStartScan?.addEventListener('click', () => {
    resetUpload();
    switchView('scan');
  });
}

// ============================================================================
// 9. UTILITY FUNCTIONS
// ============================================================================

/**
 * Formats byte count into readable KB / MB string.
 * @param {number} bytes
 * @returns {string}
 */
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/**
 * Formats an ISO timestamp into relative or short local date string.
 * @param {string} isoString
 * @returns {string}
 */
function formatRelativeTime(isoString) {
  try {
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return 'Recently';

    const diffSeconds = Math.round((Date.now() - date.getTime()) / 1000);
    if (diffSeconds < 60) return 'Just now';
    if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
    if (diffSeconds < 86400) return `${Math.floor(diffSeconds / 3600)}h ago`;
    return date.toLocaleDateString();
  } catch {
    return 'Recently';
  }
}

// ============================================================================
// 10. INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  setupProtocolAccordion();
  resetUpload();
  renderRecentScans();
  switchView('scan');

  // Initial health probe and periodic background health monitoring
  pollBackendHealth();
  state.healthTimerId = setInterval(pollBackendHealth, HEALTH_POLL_INTERVAL_MS);
});
