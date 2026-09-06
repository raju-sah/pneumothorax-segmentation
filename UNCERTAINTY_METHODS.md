# UNCERTAINTY QUANTIFICATION: Mathematical Formulations & Architectures

## 1. Mathematical Framework

Let $X \in \mathbb{R}^{H \times W \times 1}$ represent an input chest radiograph rescaled to $512 \times 512$, and let $Y \in \{0, 1\}^{H \times W}$ represent the corresponding binary segmentation ground truth.

A deep neural network parameterized by weights $\theta$ maps $X$ to pre-sigmoid logits $z(x; \theta) \in \mathbb{R}^{H \times W}$, with pixel-wise posterior probabilities:
$$\hat{p}_{ij} = \sigma(z_{ij}) = \frac{1}{1 + e^{-z_{ij}}}$$

---

## 2. Method A: Deterministic Baseline

The deterministic model trains a single parameter vector $\theta^* \approx \arg\min_\theta \mathcal{L}(\theta; \mathcal{D})$.
- **Prediction**: Single forward pass $\hat{p}(x) = \sigma(f_{\theta^*}(x))$.
- **Naive Confidence Map**: $C_{ij} = \max(\hat{p}_{ij}, 1 - \hat{p}_{ij}) \in [0.5, 1.0]$.
- **Total Uncertainty (Shannon Entropy)**:
  $$H_{\text{det}}(x)_{ij} = - \hat{p}_{ij} \log_2 \hat{p}_{ij} - (1 - \hat{p}_{ij})\log_2(1 - \hat{p}_{ij})$$
- **Limitation**: Deterministic entropy is a purely mathematical function of the predicted probability $\hat{p}$. It cannot differentiate between inherent boundary ambiguity (aleatoric noise) and model ignorance on unfamiliar artifacts (epistemic uncertainty).

---

## 3. Method B: Monte Carlo (MC) Dropout

### 3.1 Variational Formulation
Following Gal & Ghahramani (2016), placing dropout layers throughout a neural network corresponds to an approximate variational inference distribution $q(\theta)$ over network parameters.

### 3.2 Architectural Specification
- **Dropout Type**: Spatial Dropout (`nn.Dropout2d(p=0.2)`). Standard 1D dropout independently zeros single activations, which are easily reconstructed from adjacent spatial neighbors in convolutional feature maps. Spatial dropout drops entire $1 \times 1$ feature channels, forcing the network to learn spatially distributed representations.
- **Placement**: Inserted at the bottleneck and decoder stages (Blocks 1 through 4).
- **Encoder Preservation**: The ResNet34 ImageNet-pretrained encoder weights are kept dropout-free to preserve foundational visual primitives (Gabor-like edge filters, low-level radiographic contrast boundaries).

### 3.3 Test-Time Sampling & Metrics
During evaluation, dropout remains active. We draw $T = 20$ stochastic forward passes:
$$\{\hat{p}^{(t)}(x) = \sigma(f_{\tilde{\theta}_t}(x))\}_{t=1}^T, \quad \tilde{\theta}_t \sim q(\theta)$$

- **Predictive Mean**:
  $$\bar{p}_{MC}(x)_{ij} = \frac{1}{T} \sum_{t=1}^T \hat{p}^{(t)}_{ij}(x)$$
- **Epistemic Uncertainty (Predictive Variance)**:
  $$\sigma^2_{MC}(x)_{ij} = \frac{1}{T} \sum_{t=1}^T \left( \hat{p}^{(t)}_{ij}(x) - \bar{p}_{MC}(x)_{ij} \right)^2$$
- **Total Uncertainty (Predictive Entropy)**:
  $$H_{MC}(x)_{ij} = - \bar{p}_{MC}(x)_{ij} \log_2 \bar{p}_{MC}(x)_{ij} - (1 - \bar{p}_{MC}(x)_{ij}) \log_2 (1 - \bar{p}_{MC}(x)_{ij})$$

---

## 4. Method C: Deep Ensemble

### 4.1 Non-Convex Optimization Diversity
Following Lakshminarayanan et al. (2017), we train $M = 5$ identical ResNet34 U-Nets initialized with different random seeds:
$$\{\theta_m\}_{m=1}^M, \quad \theta_m \sim \text{Init}(s_m), \quad s_m \in \{42, 43, 44, 45, 46\}$$
Each ensemble member traverses an independent optimization trajectory across the non-convex empirical risk landscape, landing in distinct local minima.

### 4.2 Ensemble Metrics & Information Decomposition
For a radiograph $x$, each model outputs $\hat{p}_m(x) = \sigma(f_{\theta_m}(x))$.
- **Ensemble Mean Prediction**:
  $$\bar{p}_{\text{ens}}(x)_{ij} = \frac{1}{M} \sum_{m=1}^M \hat{p}_{m, ij}(x)$$
- **Total Uncertainty (Ensemble Entropy)**:
  $$H_{\text{total}}(x)_{ij} = H(\bar{p}_{\text{ens}}(x)_{ij})$$
- **Aleatoric Uncertainty (Expected Entropy)**:
  $$\bar{H}(x)_{ij} = \frac{1}{M} \sum_{m=1}^M H(\hat{p}_{m, ij}(x))$$
- **Epistemic Uncertainty (Mutual Information $\mathcal{I}$)**:
  $$\mathcal{I}(Y; \theta \mid X)_{ij} = H_{\text{total}}(x)_{ij} - \bar{H}(x)_{ij}$$
  Mutual information quantifies the disagreement among the ensemble members. If all models agree (even if predicting $\hat{p}=0.5$), $\mathcal{I} = 0$. If models disagree sharply (some predict $\hat{p}=1.0$, others $\hat{p}=0.0$), $\mathcal{I}$ approaches maximum value ($1.0$).
- **Ensemble Variance**:
  $$\sigma^2_{\text{ens}}(x)_{ij} = \frac{1}{M}\sum_{m=1}^M \left( \hat{p}_{m, ij}(x) - \bar{p}_{\text{ens}}(x)_{ij} \right)^2$$

---

## 5. Dense-to-Case Uncertainty Aggregation

To enable selective prediction (case-level triage), dense pixel uncertainty maps $U(x) \in \mathbb{R}^{H \times W}$ must be condensed into a single scalar confidence score $S(x) \in \mathbb{R}$:

1. **Mean Uncertainty**:
   $$S_{\text{mean}}(x) = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W U(x)_{ij}$$
2. **Top-$K$ Pixel Uncertainty ($K=500$)**:
   Extract the $K$ pixels exhibiting the highest uncertainty values:
   $$S_{\text{top-}K}(x) = \frac{1}{K} \sum_{k=1}^K U(x)_{(k)}$$
   *Clinical Rationale*: Since subtle pneumothoraces occupy a small fraction of the image area ($<2\%$), whole-image averaging dilutes localized diagnostic uncertainty with millions of certain background pixels. Top-$K$ isolates focal diagnostic ambiguity.
