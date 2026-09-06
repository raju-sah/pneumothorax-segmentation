# DATA LEAKAGE AUDIT & PREVENTION PROTOCOL

## 1. Mechanisms of Data Leakage in Medical Computer Vision

Data leakage in medical AI occurs when information from outside the training dataset is inadvertently utilized to create the model, artificially inflating performance estimates. In chest radiography, the two primary vectors of catastrophic leakage are:

1. **Patient-Level Cross-Contamination**:
   A single patient undergoes multiple radiographic examinations over hours, days, or months (e.g., initial presentation, post-thoracostomy placement, daily monitoring, follow-up after tube removal). If radiographs are randomly partitioned by `ImageId`, examinations from the same patient populate both training and test sets.
2. **Derived Dataset Overlap (The NIH ChestX-ray14 Trap)**:
   Treating another institutional dataset as an "external test cohort" when it actually shares identical source patients or images with the training dataset.

---

## 2. Patient-Level Leakage Audit in SIIM-ACR

In the 12,047-image Stage 2 dataset, extracting `PatientID` reveals that the number of unique individuals is fewer than the total number of images. Multiple images share identical `PatientID` values with differing timestamps, projections (AP and PA on the same date), or follow-up intervals.

### The Consequences of Random Image-Level Splitting
If an image-level split is used:
- The convolutional network memorizes distinctive patient-specific anatomical landmarks:
  - Specific thoracic cage geometry and rib anomalies.
  - Distinctive cardiothoracic ratio, cardiac apex contours, or pacemaker leads.
  - Chronic degenerative spinal changes, scoliosis, or osteophytes.
  - Suture wires, surgical clips, or anatomical calcifications.
- **Result**: The model achieves high test Dice by recognizing the patient's anatomy rather than learning generalizable boundary features of the visceral pleura. When deployed on unseen patients, performance plummets.

---

## 3. The NIH ChestX-ray14 Confound

> [!WARNING]
> **Definitive Finding**: The SIIM-ACR Pneumothorax Challenge dataset was constructed by sampling chest radiographs directly from the National Institutes of Health (NIH) Clinical Center ChestX-ray14 repository (Wang et al., 2017).

1. **Patient Overlap**: Evaluating a SIIM-ACR trained model on NIH ChestX-ray14 does **not** constitute external institutional validation. It evaluates on the same underlying hospital population.
2. **Label Discrepancy**: NIH ChestX-ray14 labels were generated via automated natural language processing (NLP) of text radiology reports (CheXNet), containing estimated 10% to 30% label error rates, and contains **no pixel-level pneumothorax segmentation masks**.
3. **Protocol Enforcement**: We strictly prohibit using NIH ChestX-ray14 as an external segmentation validation benchmark.

---

## 4. Leakage-Free Stratified Group Splitting Protocol

To ensure publication-grade scientific validity, data partitioning is governed by the following mathematical constraints:

### Mathematical Definition
Let $\mathcal{D} = \{(x_i, y_i, p_i)\}_{i=1}^N$ be the dataset where $x_i$ is the radiograph, $y_i$ is the binary mask, and $p_i \in \mathcal{P}$ is the unique `PatientID`.

We partition $\mathcal{P}$ into disjoint subsets:
$$\mathcal{P} = \mathcal{P}_{\text{train}} \cup \mathcal{P}_{\text{val}} \cup \mathcal{P}_{\text{test}}$$
Subject to:
$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \quad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$

Every image $i$ is assigned to the partition of its patient:
$$\mathcal{D}_{\text{split}} = \{ (x_i, y_i) \mid p_i \in \mathcal{P}_{\text{split}} \}$$

### Stratification Objectives
The grouping algorithm simultaneously balances:
1. **Patient-Level Prevalence**: The proportion of patients presenting with pneumothorax is preserved across folds:
   $$\frac{|\{p \in \mathcal{P}_{\text{split}} \mid \text{has\_pneumo}(p)\}|}{|\mathcal{P}_{\text{split}}|} \approx \text{Constant} \quad (\approx 22.15\%)$$
2. **Projection Balance**: The ratio of AP to PA projections is matched across splits.

### Verification Script Guarantee
Before running any model training, `tests/test_data_split.py` executes an automated assertion:
```python
assert len(set(train_patients).intersection(set(test_patients))) == 0, "Patient leakage detected!"
assert len(set(train_patients).intersection(set(val_patients))) == 0, "Patient leakage detected!"
assert len(set(val_patients).intersection(set(test_patients))) == 0, "Patient leakage detected!"
```
Training scripts abort immediately if any intersection is non-empty.
