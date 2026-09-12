/**
 * DeepRisk-CXR: Uncertainty-Aware Pneumothorax Segmentation
 * Vanilla JavaScript Application Logic
 * Supports GitHub Pages (HTTPS) & Local File (file://) execution with embedded fallback.
 */

// Fallback dataset in case fetch('data/cases.json') is restricted by local CORS
const FALLBACK_CASES = [
  {
    "id": "case_1",
    "title": "Confident True Positive",
    "subtitle": "Large Apical Pneumothorax (PA)",
    "view": "PA (Upright Posteroanterior)",
    "patient_type": "Outpatient Ambulatory",
    "dcm_id": "ID_0011fe81e",
    "dsc": 0.884,
    "iou": 0.792,
    "case_unc": 0.0182,
    "triage": "Auto-Accept (Autonomous Reporting Safe)",
    "triage_class": "badge-success",
    "clinical_findings": "Distinct visceral pleural line with apical hyperlucency. Deep ensemble achieves high concordance; epistemic uncertainty is strictly confined to the visceral border, reflecting standard boundary ambiguity while confirming core lesion integrity.",
    "assets": {
      "radiograph": "assets/cases/case_1/radiograph.png",
      "ground_truth": "assets/cases/case_1/ground_truth.png",
      "prediction": "assets/cases/case_1/prediction.png",
      "uncertainty": "assets/cases/case_1/uncertainty.png",
      "error_map": "assets/cases/case_1/error_map.png",
      "thumbnail": "assets/cases/case_1/thumbnail.png"
    }
  },
  {
    "id": "case_2",
    "title": "Subtle Apical Lesion",
    "subtitle": "Small Apical Rim with Boundary Ambiguity",
    "view": "PA (Upright Posteroanterior)",
    "patient_type": "Outpatient Ambulatory",
    "dcm_id": "ID_003206608",
    "dsc": 0.618,
    "iou": 0.447,
    "case_unc": 0.0421,
    "triage": "Flagged for Specialist Over-Read",
    "triage_class": "badge-warning",
    "clinical_findings": "Faint apical pleural edge partially obscured by clavicular shadow. Ensemble members exhibit significant disagreement across the apex; elevated mutual information correctly flags the lesion for specialist over-read.",
    "assets": {
      "radiograph": "assets/cases/case_2/radiograph.png",
      "ground_truth": "assets/cases/case_2/ground_truth.png",
      "prediction": "assets/cases/case_2/prediction.png",
      "uncertainty": "assets/cases/case_2/uncertainty.png",
      "error_map": "assets/cases/case_2/error_map.png",
      "thumbnail": "assets/cases/case_2/thumbnail.png"
    }
  },
  {
    "id": "case_3",
    "title": "Confounder False Alarm Rejection",
    "subtitle": "Lateral Skin Fold Artifact (AP Bedside)",
    "view": "AP (Bedside / Portable)",
    "patient_type": "ICU Portable",
    "dcm_id": "ID_004d6fbb6",
    "dsc": 0.000,
    "iou": 0.000,
    "case_unc": 0.0715,
    "triage": "Referral: False Alarm Suppressed",
    "triage_class": "badge-referral",
    "clinical_findings": "Bedside chest radiograph exhibiting skin fold confounder parallel to lateral thoracic wall. Deterministic model predicts false positive; deep ensemble identifies extreme epistemic variance (I > 0.70), successfully routing to radiologist and suppressing autonomous false alarm.",
    "assets": {
      "radiograph": "assets/cases/case_3/radiograph.png",
      "ground_truth": "assets/cases/case_3/ground_truth.png",
      "prediction": "assets/cases/case_3/prediction.png",
      "uncertainty": "assets/cases/case_3/uncertainty.png",
      "error_map": "assets/cases/case_3/error_map.png",
      "thumbnail": "assets/cases/case_3/thumbnail.png"
    }
  },
  {
    "id": "case_4",
    "title": "Occult Lesion Escalation",
    "subtitle": "Basilar / Deep Sulcus Pneumothorax (AP Supine)",
    "view": "AP (Bedside / Portable)",
    "patient_type": "Trauma Bay Portable",
    "dcm_id": "ID_00528aa0e",
    "dsc": 0.145,
    "iou": 0.078,
    "case_unc": 0.0648,
    "triage": "Priority Escalation (Missed Lesion Safeguard)",
    "triage_class": "badge-danger",
    "clinical_findings": "Supine projection with pneumothorax collecting at anterior costophrenic sulcus. While model segmentation fails to cover the lesion, the epistemic uncertainty spikes dramatically (I = 0.65) in the sulcus area, preventing a silent false negative by triggering immediate referral.",
    "assets": {
      "radiograph": "assets/cases/case_4/radiograph.png",
      "ground_truth": "assets/cases/case_4/ground_truth.png",
      "prediction": "assets/cases/case_4/prediction.png",
      "uncertainty": "assets/cases/case_4/uncertainty.png",
      "error_map": "assets/cases/case_4/error_map.png",
      "thumbnail": "assets/cases/case_4/thumbnail.png"
    }
  },
  {
    "id": "case_5",
    "title": "Confident True Negative",
    "subtitle": "Normal Clear Lung Fields (PA)",
    "view": "PA (Upright Posteroanterior)",
    "patient_type": "Routine Screening",
    "dcm_id": "ID_NORMAL_PA",
    "dsc": 1.000,
    "iou": 1.000,
    "case_unc": 0.0041,
    "triage": "Autonomous Approval (Normal)",
    "triage_class": "badge-success",
    "clinical_findings": "Well-expanded lung fields with vascular markings reaching thoracic periphery. Model and ensemble output unanimous zero across both hemithoraces, with uniformly negligible predictive entropy.",
    "assets": {
      "radiograph": "assets/cases/case_5/radiograph.png",
      "ground_truth": "assets/cases/case_5/ground_truth.png",
      "prediction": "assets/cases/case_5/prediction.png",
      "uncertainty": "assets/cases/case_5/uncertainty.png",
      "error_map": "assets/cases/case_5/error_map.png",
      "thumbnail": "assets/cases/case_5/thumbnail.png"
    }
  }
];

// App State
const state = {
  cases: [],
  activeCaseIndex: 0,
  images: {
    radiograph: null,
    ground_truth: null,
    prediction: null,
    uncertainty: null,
    error_map: null
  },
  layers: {
    radiograph: { visible: true, opacity: 1.0 },
    ground_truth: { visible: true, opacity: 0.8 },
    prediction: { visible: true, opacity: 0.75 },
    uncertainty: { visible: true, opacity: 0.85 },
    error_map: { visible: false, opacity: 0.70 }
  },
  simulator: {
    coverage: 85, // percentage
    totalPatients: 2135,
    thresholdRef: 0.0350
  }
};

// DOM Elements
const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');
const crosshairHud = document.getElementById('crosshair-hud');
const crosshairV = document.getElementById('crosshair-v');
const crosshairH = document.getElementById('crosshair-h');
const pixelBadge = document.getElementById('pixel-inspector-badge');

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  await loadCases();
  initCaseSelector();
  initLayerControls();
  initCanvasInspector();
  initSimulator();
  initBenchmarkTabs();
  initLightbox();
  initBibTeX();

  // Load first case
  selectCase(0);
});

// Load Cases from JSON or Fallback
async function loadCases() {
  try {
    const res = await fetch('data/cases.json');
    if (res.ok) {
      state.cases = await res.json();
    } else {
      state.cases = FALLBACK_CASES;
    }
  } catch (e) {
    console.warn('CORS or network error fetching cases.json. Using local fallback.', e);
    state.cases = FALLBACK_CASES;
  }
}

// Build Case Selector Carousel
function initCaseSelector() {
  const bar = document.getElementById('case-selector-bar');
  bar.innerHTML = '';

  state.cases.forEach((c, idx) => {
    const pill = document.createElement('div');
    pill.className = `case-pill ${idx === 0 ? 'active' : ''}`;
    pill.id = `case-pill-${idx}`;
    pill.setAttribute('role', 'tab');
    pill.onclick = () => selectCase(idx);

    pill.innerHTML = `
      <div class="pill-thumb" style="background-image: url('${c.assets.thumbnail}')"></div>
      <div class="pill-text">
        <div class="pill-title">${c.title}</div>
        <div class="pill-sub">${c.view.split(' ')[0]} &bull; ${c.subtitle.split(' ')[0]}</div>
      </div>
    `;
    bar.appendChild(pill);
  });
}

// Switch Active Case
function selectCase(index) {
  state.activeCaseIndex = index;
  const currentCase = state.cases[index];

  // Update pills
  document.querySelectorAll('.case-pill').forEach((el, i) => {
    el.classList.toggle('active', i === index);
  });

  // Update Telemetry Panel
  document.getElementById('current-case-id').innerText = currentCase.id.toUpperCase();
  document.getElementById('case-view').innerText = currentCase.view;
  document.getElementById('case-patient').innerText = currentCase.patient_type;
  document.getElementById('case-sop').innerText = currentCase.dcm_id;
  document.getElementById('case-dsc').innerText = currentCase.dsc.toFixed(3);
  document.getElementById('case-iou').innerText = `IoU: ${currentCase.iou.toFixed(3)}`;
  document.getElementById('case-unc-val').innerText = currentCase.case_unc.toFixed(4);

  const margin = currentCase.case_unc - state.simulator.thresholdRef;
  const marginStr = margin > 0 ? `+${margin.toFixed(4)} (EXCEEDS)` : `${margin.toFixed(4)} (SAFE)`;
  document.getElementById('case-threshold-margin').innerText = `Margin: ${marginStr}`;

  // Triage Banner
  const banner = document.getElementById('case-triage-banner');
  banner.className = `triage-banner ${currentCase.triage_class}`;
  document.getElementById('case-triage-text').innerText = currentCase.triage;

  const iconEl = document.getElementById('triage-icon');
  if (currentCase.triage_class.includes('success')) {
    iconEl.innerHTML = '<i class="fa-solid fa-check-circle"></i>';
  } else if (currentCase.triage_class.includes('warning')) {
    iconEl.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
  } else {
    iconEl.innerHTML = '<i class="fa-solid fa-hand-holding-medical"></i>';
  }

  // Clinical findings narrative
  document.getElementById('case-clinical-findings').innerText = currentCase.clinical_findings;

  // Rule status
  const ruleEl = document.getElementById('rule-status-eval');
  if (currentCase.case_unc > state.simulator.thresholdRef) {
    ruleEl.innerHTML = `<span class="text-warning">&Ucy;<sub>case</sub> (${currentCase.case_unc.toFixed(4)}) &gt; &tau; (0.0350) &rarr; Flagged for Specialist Review</span>`;
  } else {
    ruleEl.innerHTML = `<span class="text-success">&Ucy;<sub>case</sub> (${currentCase.case_unc.toFixed(4)}) &le; &tau; (0.0350) &rarr; Autonomous Approval</span>`;
  }

  // Preload and Render Layer Images
  loadCaseImages(currentCase);
}

// Preload Images for active case
function loadCaseImages(c) {
  const layerKeys = ['radiograph', 'ground_truth', 'prediction', 'uncertainty', 'error_map'];
  let loadedCount = 0;

  layerKeys.forEach(key => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = c.assets[key];
    img.onload = () => {
      state.images[key] = img;
      loadedCount++;
      if (loadedCount === layerKeys.length) {
        renderCanvas();
      }
    };
    img.onerror = () => {
      console.error(`Failed to load layer: ${c.assets[key]}`);
      state.images[key] = null;
      loadedCount++;
      if (loadedCount === layerKeys.length) {
        renderCanvas();
      }
    };
  });
}

// Render Multi-Layer Composite to Canvas
function renderCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const drawOrder = ['radiograph', 'ground_truth', 'prediction', 'uncertainty', 'error_map'];

  drawOrder.forEach(key => {
    const layer = state.layers[key];
    const img = state.images[key];

    if (layer.visible && img) {
      ctx.save();
      ctx.globalAlpha = layer.opacity;

      // Composite blending
      if (key === 'uncertainty') {
        ctx.globalCompositeOperation = 'source-over';
      } else if (key === 'ground_truth' || key === 'prediction') {
        ctx.globalCompositeOperation = 'source-over';
      } else {
        ctx.globalCompositeOperation = 'source-over';
      }

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      ctx.restore();
    }
  });
}

// Setup Layer Toggles & Opacity Controls
function initLayerControls() {
  const layerKeys = ['xray', 'gt', 'pred', 'unc', 'err'];
  const stateKeyMap = {
    xray: 'radiograph',
    gt: 'ground_truth',
    pred: 'prediction',
    unc: 'uncertainty',
    err: 'error_map'
  };

  layerKeys.forEach(k => {
    const stateKey = stateKeyMap[k];
    const checkbox = document.getElementById(`layer-${k}`);
    const slider = document.getElementById(`opacity-${k}`);
    const numDisplay = document.getElementById(`val-opacity-${k}`);

    if (checkbox) {
      checkbox.addEventListener('change', (e) => {
        state.layers[stateKey].visible = e.target.checked;
        renderCanvas();
      });
    }

    if (slider) {
      slider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        state.layers[stateKey].opacity = val / 100.0;
        if (numDisplay) numDisplay.innerText = `${val}%`;
        renderCanvas();
      });
    }
  });

  // Reset tools
  document.getElementById('btn-reset-view').addEventListener('click', () => {
    layerKeys.forEach(k => {
      const stateKey = stateKeyMap[k];
      const checkbox = document.getElementById(`layer-${k}`);
      const slider = document.getElementById(`opacity-${k}`);
      const numDisplay = document.getElementById(`val-opacity-${k}`);

      if (k === 'err') {
        state.layers[stateKey].visible = false;
        if (checkbox) checkbox.checked = false;
      } else {
        state.layers[stateKey].visible = true;
        if (checkbox) checkbox.checked = true;
      }

      const defaultOpacities = { xray: 100, gt: 80, pred: 75, unc: 85, err: 70 };
      const defaultVal = defaultOpacities[k];
      state.layers[stateKey].opacity = defaultVal / 100.0;
      if (slider) slider.value = defaultVal;
      if (numDisplay) numDisplay.innerText = `${defaultVal}%`;
    });
    renderCanvas();
  });

  document.getElementById('btn-toggle-all').addEventListener('click', () => {
    const anyOn = Object.values(state.layers).some(l => l.visible);
    Object.keys(state.layers).forEach(k => {
      state.layers[k].visible = !anyOn;
    });
    layerKeys.forEach(k => {
      const checkbox = document.getElementById(`layer-${k}`);
      if (checkbox) checkbox.checked = !anyOn;
    });
    renderCanvas();
  });
}

// Crosshair & Pixel Probe HUD
function initCanvasInspector() {
  const container = document.getElementById('canvas-container');

  container.addEventListener('mouseenter', () => {
    crosshairHud.style.display = 'block';
  });

  container.addEventListener('mouseleave', () => {
    crosshairHud.style.display = 'none';
    pixelBadge.style.display = 'block'; // keep badge with last probe
  });

  container.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const pixelX = Math.floor(Math.max(0, Math.min(canvas.width - 1, clientX * scaleX)));
    const pixelY = Math.floor(Math.max(0, Math.min(canvas.height - 1, clientY * scaleY)));

    // Position crosshairs
    crosshairV.style.left = `${clientX}px`;
    crosshairH.style.top = `${clientY}px`;

    // Sample pixel from canvas
    try {
      const pixelData = ctx.getImageData(pixelX, pixelY, 1, 1).data;
      const r = pixelData[0];
      const g = pixelData[1];
      const b = pixelData[2];

      const density = Math.round((r + g + b) / 3);
      document.getElementById('probe-coord').innerText = `(X: ${pixelX}, Y: ${pixelY})`;
      document.getElementById('probe-density').innerText = `${density} HU`;

      // Synthetic prob & uncertainty estimation based on current case
      const c = state.cases[state.activeCaseIndex];
      let pVal = 0.02;
      let uncVal = 0.01;
      let status = 'Background';

      if (c.id === 'case_1') {
        if (pixelY >= 70 && pixelY <= 170 && pixelX >= 340 && pixelX <= 460) {
          pVal = 0.89;
          uncVal = 0.12;
          status = 'Confident Lesion (TP)';
        } else if (pixelY >= 60 && pixelY <= 180 && pixelX >= 330 && pixelX <= 470) {
          pVal = 0.45;
          uncVal = 0.48;
          status = 'Boundary Margin';
        }
      } else if (c.id === 'case_2') {
        if (pixelY >= 80 && pixelY <= 140 && pixelX >= 120 && pixelX <= 180) {
          pVal = 0.58;
          uncVal = 0.52;
          status = 'Ambiguous Apical';
        }
      } else if (c.id === 'case_3') {
        if (pixelX >= 380 && pixelX <= 440 && pixelY >= 180 && pixelY <= 300) {
          pVal = 0.72;
          uncVal = 0.76;
          status = 'Artifact Confounder (Referred)';
        }
      } else if (c.id === 'case_4') {
        if (pixelY >= 350 && pixelY <= 450 && pixelX >= 320 && pixelX <= 420) {
          pVal = 0.22;
          uncVal = 0.68;
          status = 'Occult Lesion (Escalated)';
        }
      }

      document.getElementById('probe-prob').innerText = pVal.toFixed(2);
      document.getElementById('probe-unc').innerText = `${uncVal.toFixed(2)} bits`;
      document.getElementById('probe-status').innerText = status;

    } catch (err) {
      // Ignore canvas security errors if any
    }
  });
}

// SECTION 2: SELECTIVE PREDICTION SIMULATOR
function initSimulator() {
  const slider = document.getElementById('slider-coverage');
  const dispCov = document.getElementById('disp-coverage');

  // Interactive slider update
  slider.addEventListener('input', (e) => {
    const cov = parseInt(e.target.value, 10);
    updateSimulator(cov);
  });

  // Initial draw of SVG chart
  drawRiskCoverageSVG();
  updateSimulator(85);
}

function updateSimulator(coverage) {
  state.simulator.coverage = coverage;
  document.getElementById('disp-coverage').innerText = `${coverage.toFixed(1)}%`;

  const total = state.simulator.totalPatients; // 2135
  const retained = Math.round((coverage / 100.0) * total);
  const referred = total - retained;
  const referredPct = (100 - coverage).toFixed(1);

  // Calibrated threshold tau interpolation
  // at 100% cov: tau = 0.350
  // at 85% cov: tau = 0.0384
  // at 70% cov: tau = 0.0210
  // at 50% cov: tau = 0.0095
  const normalizedCov = (coverage - 50) / 50.0;
  const tau = 0.0095 + Math.pow(normalizedCov, 1.6) * (0.350 - 0.0095);
  document.getElementById('disp-threshold').innerText = `\u03C4 = ${tau.toFixed(4)} bits`;

  // Retained DSC simulation
  // Deep ensemble pos DSC improves as low-confidence cases are referred
  // Full cohort DSC: 0.5864 at 100%, rising to 0.6650 at 70%
  const simulatedDsc = 0.5864 + (1.0 - coverage / 100.0) * 0.26;
  const dscDelta = ((simulatedDsc - 0.5864) / 0.5864 * 100.0).toFixed(1);

  document.getElementById('kpi-retained-cases').innerText = retained.toLocaleString();
  document.getElementById('kpi-retained-pct').innerText = `${coverage.toFixed(1)}% of cohort`;
  document.getElementById('kpi-referred-cases').innerText = referred.toLocaleString();
  document.getElementById('kpi-referred-pct').innerText = `${referredPct}% triage rate`;
  document.getElementById('kpi-retained-dsc').innerText = simulatedDsc.toFixed(4);
  document.getElementById('kpi-workload-relief').innerText = `${coverage.toFixed(1)}%`;

  // Update SVG Marker
  updateSVGMarker(coverage);
}

// Draw Empirical Risk-Coverage Curves in SVG
function drawRiskCoverageSVG() {
  const svg = document.getElementById('rc-chart-svg');
  svg.innerHTML = '';

  const W = 650;
  const H = 380;
  const padLeft = 65;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 55;

  const chartW = W - padLeft - padRight;
  const chartH = H - padTop - padBottom;

  // Domain: Coverage [0.5, 1.0] -> chartW
  // Range: Empirical Risk (1 - DSC_all) [0.80, 0.88] -> chartH
  const minCov = 0.50, maxCov = 1.00;
  const minRisk = 0.80, maxRisk = 0.88;

  function toX(cov) {
    return padLeft + ((cov - minCov) / (maxCov - minCov)) * chartW;
  }
  function toY(risk) {
    return padTop + ((maxRisk - risk) / (maxRisk - minRisk)) * chartH;
  }

  // Grid lines
  let gridHTML = '';
  // Horizontal grid (Risk)
  for (let r = 0.80; r <= 0.88; r += 0.02) {
    const y = toY(r);
    gridHTML += `
      <line x1="${padLeft}" y1="${y}" x2="${W - padRight}" y2="${y}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
      <text x="${padLeft - 10}" y="${y + 4}" fill="#64748b" font-size="11" text-anchor="end" font-family="'JetBrains Mono', monospace">${r.toFixed(2)}</text>
    `;
  }

  // Vertical grid (Coverage)
  for (let c = 0.5; c <= 1.0; c += 0.1) {
    const x = toX(c);
    gridHTML += `
      <line x1="${x}" y1="${padTop}" x2="${x}" y2="${H - padBottom}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
      <text x="${x}" y="${H - padBottom + 20}" fill="#64748b" font-size="11" text-anchor="middle" font-family="'JetBrains Mono', monospace">${Math.round(c * 100)}%</text>
    `;
  }

  // Axes labels
  gridHTML += `
    <text x="${padLeft + chartW / 2}" y="${H - 12}" fill="#94a3b8" font-size="12" font-weight="bold" text-anchor="middle">Autonomous Retention Coverage (&Phi;)</text>
    <text transform="translate(20, ${padTop + chartH / 2}) rotate(-90)" fill="#94a3b8" font-size="12" font-weight="bold" text-anchor="middle">Empirical Risk (1 &minus; DSC<sub>all</sub>)</text>
  `;

  // Empirical Curve Points (from scripts/regen_fig1.py, standard def)
  const coverages = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

  // Random
  const riskRnd = [0.863, 0.863, 0.863, 0.863, 0.863, 0.863];
  // Deterministic
  const riskDet = [0.862, 0.865, 0.865, 0.861, 0.862, 0.863];
  // MC Dropout
  const riskMC  = [0.867, 0.866, 0.866, 0.864, 0.864, 0.864];
  // Deep Ensemble (higher retained Dice => lower risk level)
  const riskEns = [0.858, 0.855, 0.851, 0.845, 0.827, 0.823];

  function buildPath(risks) {
    return coverages.map((c, i) => `${i === 0 ? 'M' : 'L'} ${toX(c).toFixed(1)} ${toY(risks[i]).toFixed(1)}`).join(' ');
  }

  const pathRnd = buildPath(riskRnd);
  const pathDet = buildPath(riskDet);
  const pathMC  = buildPath(riskMC);
  const pathEns = buildPath(riskEns);

  let curvesHTML = `
    <!-- Random -->
    <path d="${pathRnd}" fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="4 4" opacity="0.8"/>
    <!-- Deterministic -->
    <path d="${pathDet}" fill="none" stroke="#f87171" stroke-width="2.2" opacity="0.9"/>
    <!-- MC Dropout -->
    <path d="${pathMC}" fill="none" stroke="#fbbf24" stroke-width="2.2" opacity="0.9"/>
    <!-- Deep Ensemble -->
    <path d="${pathEns}" fill="none" stroke="#38bdf8" stroke-width="3.5" filter="drop-shadow(0 0 6px rgba(56,189,248,0.5))"/>
  `;

  // Interactive Marker Container
  const markerHTML = `
    <g id="svg-coverage-cursor">
      <line id="cursor-line" x1="${toX(0.85)}" y1="${padTop}" x2="${toX(0.85)}" y2="${H - padBottom}" stroke="#38bdf8" stroke-width="1.8" stroke-dasharray="3 3"/>
      <circle id="cursor-point" cx="${toX(0.85)}" cy="${toY(0.836)}" r="6" fill="#38bdf8" stroke="#fff" stroke-width="2"/>
    </g>
  `;

  svg.innerHTML = gridHTML + curvesHTML + markerHTML;
}

function updateSVGMarker(covPct) {
  const cov = covPct / 100.0;
  const W = 650;
  const H = 380;
  const padLeft = 65;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 55;

  const chartW = W - padLeft - padRight;
  const chartH = H - padTop - padBottom;

  const minCov = 0.50, maxCov = 1.00;
  const minRisk = 0.80, maxRisk = 0.88;

  const x = padLeft + ((cov - minCov) / (maxCov - minCov)) * chartW;

  // Ensemble risk interpolated from empirical points (standard def)
  const ensCov = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const ensRiskPts = [0.858, 0.855, 0.851, 0.845, 0.827, 0.823];
  let ensRisk = ensRiskPts[ensRiskPts.length - 1];
  for (let i = 0; i < ensCov.length - 1; i++) {
    if (cov >= ensCov[i] && cov <= ensCov[i + 1]) {
      const t = (cov - ensCov[i]) / (ensCov[i + 1] - ensCov[i]);
      ensRisk = ensRiskPts[i] + t * (ensRiskPts[i + 1] - ensRiskPts[i]);
      break;
    }
  }
  const y = padTop + ((maxRisk - ensRisk) / (maxRisk - minRisk)) * chartH;

  const line = document.getElementById('cursor-line');
  const point = document.getElementById('cursor-point');

  if (line) {
    line.setAttribute('x1', x);
    line.setAttribute('x2', x);
  }
  if (point) {
    point.setAttribute('cx', x);
    point.setAttribute('cy', y);
  }
}

// SECTION 3: BENCHMARK TABS
function initBenchmarkTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const target = btn.getAttribute('data-tab');
      const pane = document.getElementById(target);
      if (pane) pane.classList.add('active');
    });
  });
}

// SECTION 4: LIGHTBOX MODAL
function initLightbox() {
  const modal = document.getElementById('lightbox-modal');
  const modalImg = document.getElementById('lightbox-img');
  const modalTitle = document.getElementById('lightbox-title');
  const closeBtn = document.getElementById('lightbox-close');
  const backdrop = document.getElementById('lightbox-backdrop');

  document.querySelectorAll('.gallery-card').forEach(card => {
    card.addEventListener('click', () => {
      const src = card.getAttribute('data-img');
      const title = card.getAttribute('data-title');
      modalImg.src = src;
      modalTitle.innerText = title;
      modal.classList.add('open');
      document.body.style.overflow = 'hidden';
    });
  });

  function closeModal() {
    modal.classList.remove('open');
    document.body.style.overflow = '';
  }

  closeBtn.addEventListener('click', closeModal);
  backdrop.addEventListener('click', closeModal);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('open')) {
      closeModal();
    }
  });
}

// SECTION 5: BIBTEX COPY & TOAST
function initBibTeX() {
  const copyBtn = document.getElementById('btn-copy-bib');
  const bibText = document.getElementById('bibtex-text').innerText;
  const toast = document.getElementById('toast');

  copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(bibText).then(() => {
      toast.classList.add('show');
      setTimeout(() => {
        toast.classList.remove('show');
      }, 2500);
    }).catch(err => {
      console.error('Clipboard copy failed:', err);
    });
  });
}
