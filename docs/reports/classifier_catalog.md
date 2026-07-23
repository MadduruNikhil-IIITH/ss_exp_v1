# SQuAD Sentence Salience - Classifier Catalog

This document catalogs all **10 core model configurations** evaluated in our query-independent experiments, detailing their architectures, feature dimensions, and core parameters.

---

## 1. Heuristic Baseline

### Config 1: RST Rule-Based
* **Architecture**: A rule-based classifier that maps rhetorical structure tree nodes to salience scores without learnable weights.
* **Mechanism**: Scores sentences based on whether they contain document RST root nodes, are marked as nuclei, and reside at shallow depths in the tree.
* **Parameters**: Decision threshold is fixed. Output is a probability-like score.

---

## 2. Logistic Regression Classifiers

All Logistic Regression configurations use `sklearn.linear_model.LogisticRegression` with a standard scaler, `max_iter=1000`, and `class_weight='balanced'`.

### Config 2: LR (Rst)
* **Inputs**: 18 Rhetorical Structure Theory (RST) features.
* **Dimensionality**: 18-dimensional vector.

### Config 3: LR (Linguistic)
* **Inputs**: 35 surface linguistic, syntactic, and readability features.
* **Dimensionality**: 35-dimensional vector.

### Config 4: LR (Surprisal)
* **Inputs**: 17 context-aware word surprisal features from PsychFormers (Causal + PLL) and GPT-2 Deletion Drop.
* **Dimensionality**: 17-dimensional vector.

### Config 5: LR (Combined)
* **Inputs**: All 70 baseline tabular features.
* **Dimensionality**: 70-dimensional vector.

---

## 3. Hybrid BERT Classifiers

All BERT-based classifiers use a frozen `bert-base-uncased` backbone (768 hidden dimensions) and train their respective fusion and classification heads.

### Config 6: Gated BERT (Context Features)
* **Architecture**: Gated vector fusion. Project the 70-dimensional tabular features to 768 dimensions. Use a sigmoid gate layer to perform element-wise fusion of the BERT embedding and projected features:
  $$\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{h}_{\text{bert}} ; \mathbf{h}_{\text{tab}}] + \mathbf{b}_g)$$
  $$\mathbf{h}_{\text{combined}} = \mathbf{g} \odot \mathbf{h}_{\text{bert}} + (1 - \mathbf{g}) \odot \mathbf{h}_{\text{tab}}$$
* **Classifier**: Single linear layer mapping $\mathbf{h}_{\text{combined}} \to 1$.

### Config 7: FiLM BERT (RST Modulation)
* **Architecture**: Feature-conditioned modulation (FiLM). The 18 RST features generate scale ($\boldsymbol{\gamma}$) and shift ($\boldsymbol{\beta}$) vectors that modulate the BERT embedding:
  $$\mathbf{h}_{\text{modulated}} = \boldsymbol{\gamma} \odot \mathbf{h}_{\text{bert}} + \boldsymbol{\beta}$$
  To prevent the model from ignoring tabular features, raw RST and other tabular features are concatenated as a skip link directly to the classification layer.
* **Classifier**: Linear layer mapping $[\mathbf{h}_{\text{modulated}} ; \mathbf{x}_{\text{rst}} ; \mathbf{x}_{\text{other}}] \to 1$.

### Config 8: Concat BERT (Context Features)
* **Architecture**: Projects the 70 tabular features to 768 dimensions and concatenates them directly with the BERT CLS token embedding.
* **Classifier**: Linear layer mapping $[\mathbf{h}_{\text{bert}} ; \mathbf{h}_{\text{tab}}] \to 1$.

### Config 9: Gated BERT (No RST)
* **Architecture**: Identical to Config 6 (Gated vector fusion), but the 18 RST features are completely excluded from the tabular input.
* **Dimensionality**: 52-dimensional tabular feature vector.

### Config 11: Heuristic-Guided BERT (RST Prior)
* **Architecture**: Evaluates sentence salience using standard BERT text embeddings, and fuses the Rule-Based RST Scorer (Config 1) probability directly into the final classification head with a learnable scalar weight:
  $$\text{logit}(s) = \mathbf{w}_{\text{bert}}^T \mathbf{h}_{\text{bert}} + w_h \cdot h(s) + b$$
* **Parameters**: Fits $w_h$ (heuristic weight) and bias $b$ on the training data.
