# SQuAD Sentence Salience - Data Balancing Process

This document details the dataset balancing techniques implemented in `src/data_processing.py` to address class imbalance in sentence salience classification.

---

## 1. The Class Imbalance Problem

In SQuAD sentence salience, a context passage typically has 5–8 sentences, but exactly **one** sentence contains the answer to the question (Class 1, salient). The rest are Class 0 (non-salient).
* **Imbalance Ratio**: ~1:4.09 (19.64% positive examples in the training set).
* **The Risk**: Models trained without balancing tend to predict Class 0 by default, or exploit simple shortcuts like sentence length or positional biases (since early sentences are more likely to contain answers in SQuAD).

To address this, we compare five distinct training-only balancing methods.

---

## 2. The Five Balancing Methods

### 1. None (Baseline Unbalanced)
* **Description**: No balancing is applied; models are trained on the natural SQuAD distribution.
* **Mathematical Formulation**: Prior class probability $\hat{P}(Y=1) \approx 0.1572$.
* **Analysis**: Matches the validation set distribution perfectly. This keeps the classification decision boundary aligned with the default $0.5$ probability threshold, leading to high raw Accuracy and Precision, but makes the model vulnerable to positional and keyword biases.

### 2. Pairwise Balancing (RankNet)
* **Description**: Transforms pointwise classification into a relative ranking problem by generating pairs of salient ($s^+$) and non-salient ($s^-$) sentences from the same context.
* **Mathematical Formulation**:
  * For each pair $(s^+, s^-)$, the input vector is the difference: $\mathbf{x}_{\text{pair}} = \mathbf{x}_{s^+} - \mathbf{x}_{s^-}$.
  * The label $y$ is set to $1$ (if $s^+$ is ordered first) or $0$ (if we swap the order, keeping labels exactly balanced at 50/50).
  * Loss is optimized via: $\text{Loss} = \text{BCEWithLogitsLoss}(\text{logit}(s^+) - \text{logit}(s^-), y)$.
* **Analysis**: Extremely powerful for ranking (high MRR/NDCG). However, because the loss depends only on logit differences, the absolute logit scale is shift-invariant, leading to uncalibrated probability thresholds when evaluated pointwise.

### 3. Cluster-Based Undersampling
* **Description**: Clusters the majority class (Class 0, non-salient sentences) using K-Means into $K$ partitions (where $K$ is the number of salient sentences). Only the representative sentence closest to each cluster centroid is kept.
* **Mathematical Formulation**:
  * Partition negative vectors $\{\mathbf{x}_i^-\}$ into $K$ clusters $C_1, C_2, \dots, C_K$ by minimizing:
    $$\arg\min_C \sum_{j=1}^K \sum_{\mathbf{x} \in C_j} \|\mathbf{x} - \boldsymbol{\mu}_j\|^2$$
  * For each cluster $C_j$, select the representative index:
    $$i_j^* = \arg\min_i \|\mathbf{x}_i^- - \boldsymbol{\mu}_j\|$$
* **Analysis**: Reduces the dataset size while preserving the feature space variance. However, K-means clusters partition sentences along simple syntactic or length properties, making training artificially simple. This leads models to overfit on simple boundaries, causing poor validation performance.

### 4. RST-Neighborhood Undersampling
* **Description**: A discourse-aware local neighborhood selection method. For each positive sentence, we select the negative sentences that are structurally closest in the document's Rhetorical Structure Theory (RST) tree and linear position.
* **Mathematical Formulation**:
  * Let $s_i^+$ be a salient sentence at index $i$ with RST tree depth $d(s_i^+)$.
  * The distance to a negative candidate $s_j^-$ is:
    $$\text{Dist}(s_i^+, s_j^-) = 0.5 \cdot |i - j| + 0.5 \cdot |d(s_i^+) - d(s_j^-)|$$
  * We select the $K$ negatives that minimize this distance.
* **Analysis**: Ensures that negatives are drawn from the same discourse paragraph neighborhood as the answers, forcing the model to learn localized structural features rather than global topic shifts.

### 5. DSNB (Discourse-Semantic Neighborhood Balancing)
* **Description**: Our main proposed balancing method. For each salient sentence, we mine the "hardest" negatives within its local neighborhood using a hybrid metric that combines linear proximity, semantic question relevance (SBERT), and discourse depth similarity.
* **Mathematical Formulation**:
  * Let $s_j^-$ be a negative candidate and $s_i^+$ be a salient sentence.
  * We compute three weights:
    1. **Positional Proximity**: $w_{\text{pos}} = 0.5^{|i - j|}$ (exponential decay with distance).
    2. **Semantic Alignment**: $w_{\text{sem}} = \max(0, \cos(\mathbf{q}, \mathbf{s}_j^-))$ (SBERT cosine similarity to the question $\mathbf{q}$).
    3. **RST Depth Similarity**: $w_{\text{rst}} = 1.0 - \frac{|d(s_i^+) - d(s_j^-)|}{\max(d)}$ (clamped to $[0, 1]$).
  * The hardness score for $s_j^-$ is:
    $$\text{Hardness}(s_j^-) = 0.4 \cdot w_{\text{pos}} + 0.4 \cdot w_{\text{sem}} + 0.2 \cdot w_{\text{rst}}$$
  * We select the negative sentences with the highest hardness scores.
* **Analysis**: DSNB constructs a training set composed of hard negatives that look like answers (high semantic similarity) and occupy similar discourse structures (RST depth). This forces the classifier to look beyond simple keyword matching and learn fine-grained discourse boundaries, resulting in highly robust model generalizability.

---

## 3. Dataset Dimensions per Balancing Method

The table below outlines the exact number of training records and class allocations after applying each balancing method:

| Balancing Method | Training Samples | Salient (Class 1) | Non-Salient (Class 0) | Label Balance |
| :--- | :---: | :---: | :---: | :---: |
| **None (Unbalanced Raw)** | 2,301 | 452 | 1,849 | 19.64% / 80.36% |
| **Pairwise** | 1,849 | 925 | 924 | 50.00% / 50.00% |
| **Cluster** | 892 | 446 | 446 | 50.00% / 50.00% |
| **RST-Neighborhood** | 892 | 446 | 446 | 50.00% / 50.00% |
| **DSNB (Proposed)** | 892 | 446 | 446 | 50.00% / 50.00% |

