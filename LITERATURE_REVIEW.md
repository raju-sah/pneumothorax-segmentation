# LITERATURE REVIEW: Uncertainty-Aware Medical Segmentation & Selective Prediction

## 1. Pneumothorax Segmentation on Chest Radiographs

### 1.1 Clinical Significance & Radiographic Hallmarks
Pneumothorax is characterized by the entry of air into the pleural cavity, breaking the negative intrapleural pressure that maintains lung inflation. Detecting pneumothorax on standard posteroanterior (PA) or anteroposterior (AP) chest radiographs is among the most critical tasks in emergency thoracic radiology.
- **The Visceral Pleural White Line**: The definitive radiographic finding is a fine, sharply defined white visceral pleural line separated from the chest wall by an avascular space devoid of bronchovascular lung markings (Rao et al., 2020).
- **Pitfalls & Pseudopneumothoraces**: Radiologists frequently encounter mimics such as skin folds, scapular borders, companion shadows of the ribs, bullae, elevated diaphragms, and clothing artifacts. Distinguishing these from subtle apical pneumothoraces requires expert visual resolution.

### 1.2 Deep Learning Paradigms & SIIM-ACR Benchmark
The 2019 Society for Imaging Informatics in Medicine (SIIM) and American College of Radiology (ACR) challenge accelerated deep learning solutions:
- **Top Solutions**: The winning team (*Bestfitting*) employed an ensemble of diverse architectures (SE-ResNeXt-101, SENet-154, EfficientNet-B4) paired with feature pyramid decoders and heavy augmentation.
- **Two-Stage Cascades**: Multiple competitors established that directly training a single segmentation model on the raw dataset resulted in high false-positive rates due to the ~78% negative case imbalance. Introducing a binary classification filter (pneumothorax present vs. absent) prior to segmentation yielded significant leaderboard gains.
- **Recent Trends (2021–2025)**: Transformer-based segmentation (e.g., TransUNet, Chen et al., 2021; Swin-UNet, Cao et al., 2022) demonstrated competitive overlap metrics but required massive pretraining datasets and incurred high GPU memory consumption.

---

## 2. Uncertainty Quantification in Medical Image Segmentation

### 2.1 Taxonomy: Aleatoric vs. Epistemic Uncertainty
Following Kendall & Gal (2017), predictive uncertainty decomposes into:
1. **Aleatoric Uncertainty**: Inherent stochasticity and noise in the data acquisition process (e.g., patient motion blur, low radiation dose, overlapping bone density, anatomical ambiguity along boundary edges). Cannot be reduced by collecting more training data of the same type.
2. **Epistemic Uncertainty**: Ignorance in model parameters due to limited or non-representative training data. Elevated in rare pathologies, out-of-distribution artifacts, and unseen patient demographics. Can theoretically be reduced with additional diverse data.

### 2.2 Monte Carlo (MC) Dropout
Introduced by Gal & Ghahramani (2016), MC Dropout casts standard dropout regularization as an approximate variational inference method for Deep Gaussian Processes. By retaining dropout at inference time and drawing $T$ stochastic samples:
$$\bar{p}(x) = \frac{1}{T}\sum_{t=1}^T \hat{p}_t(x), \quad \sigma^2(x) = \frac{1}{T}\sum_{t=1}^T (\hat{p}_t(x) - \bar{p}(x))^2$$
- **Advantages**: Requires training only a single model; readily integrated into existing architectures.
- **Limitations in Segmentation**: Standard dropout disrupts spatial structure. As demonstrated by Lambert et al. (2024), MC Dropout in medical imaging frequently produces overconfident, narrow predictive distributions that poorly correlate with true spatial boundary error unless spatial dropout (Tompson et al., 2015) is placed specifically in late decoder stages.

### 2.3 Deep Ensembles
Formalized by Lakshminarayanan et al. (2017), Deep Ensembles train $M$ identical neural networks initialized from different random seeds on shuffled batches:
- In non-convex optimization, distinct random initializations navigate the network toward functionally diverse local minima in the parameter space.
- In rigorous empirical benchmarks across computer vision and medical imaging (Ovadia et al., 2019; Beluch et al., 2018), Deep Ensembles consistently outperform Bayesian approximations (including MC Dropout and Bayes-by-Backprop) in calibration, out-of-distribution detection, and epistemic uncertainty quality.

### 2.4 Why Probabilistic U-Net is Unsuitable for this Task
The Probabilistic U-Net (Kohl et al., 2018) couples a conditional variational autoencoder (cVAE) with a U-Net to learn a continuous latent space over segmentations. While effective when multiple ground-truth annotations from different radiologists exist per image (such as LIDC-IDRI lung nodules):
1. **Class Imbalance & Posterior Collapse**: When trained on datasets with ~78% negative cases and thin hairline masks, the latent space frequently suffers from posterior collapse (where the posterior collapses to the prior, generating identical mean predictions).
2. **Hyperparameter Fragility**: Tuning the KL-divergence weight $\beta$ and annealing schedules requires excessive compute trials incompatible with part-time constraints.

---

## 3. Calibration in Semantic Segmentation

A model is calibrated if its predicted probability reflects empirical ground-truth accuracy:
$$P(\hat{Y} = Y \mid \hat{P} = p) = p, \quad \forall p \in [0, 1]$$
- Modern deep neural networks trained with cross-entropy and soft Dice are notoriously overconfident (Guo et al., 2017).
- In medical segmentation, Expected Calibration Error (ECE) is adapted to pixel spaces as **Expected Segmentation Calibration Error (ESCE)**.
- Calibration is evaluated alongside the **Brier Score**, a strictly proper scoring rule that penalizes both lack of sharpness and miscalibration.

---

## 4. Selective Prediction & Clinical Referral

### 4.1 The Risk-Coverage Paradigm
In safety-critical clinical environments, autonomous full-automation is often unacceptable. Selective prediction (Geifman & El-Yaniv, 2017; Mozannar & Sontag, 2020) provides a formal mathematical framework for automated triage:
- The model outputs a prediction $\hat{y}$ and a confidence score $g(x)$.
- A selection function $g(x) \ge \tau$ determines whether the model accepts the case or defers to a human physician.
- By varying the threshold $\tau$, an empirical **Risk-Coverage (RC) Curve** is constructed, plotting error (risk) as a function of retained cases (coverage).
- **Area Under the Risk-Coverage Curve (AURC)** serves as the canonical summary metric; lower AURC indicates superior referral efficiency.

### 4.2 Application to Chest Radiography Triage
While selective classification has been explored for disease categorization (e.g., normal vs. abnormal CXRs in Raghu et al., 2019), selective prediction for **dense semantic segmentation of subtle thoracic emergencies** remains largely unaddressed. This project bridges that gap by establishing aggregated uncertainty metrics as practical referral thresholds.
