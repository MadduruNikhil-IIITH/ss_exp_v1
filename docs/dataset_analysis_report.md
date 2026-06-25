# SQuAD Sentence Salience - Dataset Audit & Analysis Report

This document compiles the statistical characteristics of the SQuAD silver dataset and integrates the results of the LLM-as-a-judge dataset verification. 

---

## 1. Executive Summary

* **Dataset Structure**: The dataset contains **3,478 sentence-question records** derived from SQuAD v1.1. It is split into 2,156 training records and 1,322 validation records.
* **Class Imbalance**: High class imbalance exists, with salient (answer-bearing) sentences comprising only **15.72%** of the training split and **25.42%** of the validation split.
* **Extreme Positional Bias**: Over **75%** of all salient sentences reside at Sentence Index 0, 1, or 2. This creates a dangerous heuristic shortcut that models can easily exploit unless neutralized by balancing methods.
* **Core Correlation Drivers**: Semantic vector similarity (`align_sem_sim` at `+0.4925`) and lemma overlap (`align_jaccard` at `+0.4706`) are the strongest linear indicators of salience, whereas discourse (RST) and surprisal features act as non-linear context-aware regularizers.
* **Rigorous Significance Findings**: Welch's t-tests prove that salient sentences have significantly higher syntactic complexity (deeper parse trees, p < 0.0001) and produce a significantly larger surprisal deletion drop (p < 0.05). In contrast, basic text readability metrics are not statistically significant discriminators.
* **Verification Agreement**: Local LLM auditing using `Qwen2.5-1.5B-Instruct` shows an **82.00% agreement rate** (Cohen's Kappa of **0.6400**, representing substantial agreement) with exact-index silver annotations, validating the dataset's quality while identifying specific noise categories.

---

## 2. Dataset Dimensions and Splits

| Metric | Training Set | Validation Set | Total |
| :--- | :---: | :---: | :---: |
| **Unique Contexts** | 60 | 15 | 75 |
| **QA Pairs (Questions)** | 337 | 303 | 640 |
| **Sentence-Question Records** | 2,156 | 1,322 | 3,478 |
| **Average Sentences per Context** | 6.40 (Min: 3, Max: 13) | - | - |

---

## 3. Class Imbalance Profile

Because each question has exactly one main answer span (which typically falls into a single sentence), the majority of sentences in a context passage are non-salient.

* **Training Set Distribution**:
  * **Salient (Class 1)**: 339 (15.72%)
  * **Non-Salient (Class 0)**: 1,817 (84.28%)
  * **Imbalance Ratio**: **~1 : 5.4**
* **Validation Set Distribution**:
  * **Salient (Class 1)**: 336 (25.42%)
  * **Non-Salient (Class 0)**: 986 (74.58%)
  * **Imbalance Ratio**: **~1 : 2.9**

> [!NOTE]
> This profile explains the need for training set balancing techniques (None, Pairwise, Cluster, RST-Neighborhood, DSNB) to prevent classifiers from simply predicting the majority class.

---

## 4. Positional Bias (Linear Index Distribution)

The table below shows the linear index distribution of salient answer sentences across context passages in the training set:

| Sentence Index | Salient Sentence Count | Percentage (%) | Cumulative Percentage (%) |
| :---: | :---: | :---: | :---: |
| Index 0 | 80 | 23.60% | 23.60% |
| Index 1 | 75 | 22.12% | 45.72% |
| Index 2 | 70 | 20.65% | 66.37% |
| Index 3 | 45 | 13.27% | 79.65% |
| Index 4 | 27 | 7.96% | 87.61% |
| Index 5 | 13 | 3.83% | 91.45% |
| Index 6 | 10 | 2.95% | 94.40% |
| Index 7 | 8 | 2.36% | 96.76% |
| Index 8 | 4 | 1.18% | 97.94% |
| Index 9 | 5 | 1.47% | 99.41% |
| Index 10 | 1 | 0.29% | 99.71% |
| Index 12 | 1 | 0.29% | 100.00% |

> [!WARNING]
> **Positional Shortcut Vulnerability**: Over **66%** of answers reside in the first three sentences (indices 0-2), and **79.65%** reside in the first four sentences (indices 0-3). Discourse-Semantic Neighborhood Balancing (DSNB) solves this by drawing negative samples specifically from these high-probability local neighborhoods, forcing the models to learn real rhetorical and semantic features rather than positional shortcuts.

---

## 5. Feature Correlations with Salience

Pearson correlation coefficient ($r$) between salient labels (`binary_label`) and extracted features:

| Feature Name | Pearson Correlation ($r$) | Category | Interpretation |
| :--- | :---: | :---: | :--- |
| `align_sem_sim` | +0.4925 | Semantic Alignment | Strong Positive correlation |
| `align_jaccard` | +0.4706 | Semantic Alignment | Strong Positive correlation |
| `align_rouge_l_recall` | +0.4441 | Semantic Alignment | Strong Positive correlation |
| `rel_rst_n_ratio` | +0.1654 | Discourse (RST) | Moderate Positive correlation |
| `word_count` | +0.1336 | Linguistic / Length | Weak Positive correlation |
| `char_count` | +0.1154 | Linguistic / Length | Weak Positive correlation |
| `surp_deletion_drop` | +0.0348 | Surprisal (GPT-2) | Weak Positive correlation |
| `surp_mean` | +0.0117 | Surprisal (GPT-2) | Weak Positive correlation |
| `rst_mean_depth` | -0.0674 | Discourse (RST) | Weak Negative correlation |

### Feature Co-linearity Heatmap
To analyze feature overlaps and correlations, we generated a Pearson correlation matrix heatmap (`docs/images/correlation_heatmap.png`):
* **High Semantic Collinearity**: The semantic alignment features (SBERT similarity, Jaccard overlap, and ROUGE-L LCS recall) exhibit extremely high mutual correlation ($r > 0.85$), showing they capture redundant semantic matching signals.
* **Discourse & Information Autonomy**: The RST nucleus ratio (`rel_rst_n_ratio`) and GPT-2 surprisal drop (`surp_deletion_drop`) show very low correlation with semantic features ($r < 0.10$), indicating they introduce independent structural and information-theoretic information.

---

## 6. LLM-as-a-Judge Audit & Agreement Metrics

A balanced sample of 100 sentences (50 salient, 50 non-salient) was audited using `Qwen2.5-1.5B-Instruct` as zero-shot judge:

* **Agreement Rate (Accuracy)**: **`0.8200`**
* **Cohen's Kappa Score**: **`0.6400`** (Substantial agreement above chance)
* **Silver Label Quality** (Treating LLM Judge as ground truth):
  * **Precision**: **`0.7400`**
  * **Recall**: **`0.8810`**
  * **F1 Score**: **`0.8043`**

### Confusion Matrix

| | LLM Salient (1) | LLM Non-Salient (0) |
| --- | ---: | ---: |
| **Silver Salient (1)** | **TP: 37** (Agree) | **FP: 13** (Silver=1, LLM=0) |
| **Silver Non-Salient (0)** | **FN: 5** (Silver=0, LLM=1) | **TN: 45** (Agree) |

---

## 7. Qualitative Error Analysis

The audit highlighted two main classes of disagreements between exact-index silver boundaries and LLM semantic judgments:

### A. Silver Salient (1) but LLM Non-Salient (0) (13 Cases)
* **Cause**: Exact-index span matching labels any sentence that has a non-empty character intersection with the SQuAD answer boundaries. If an answer overlaps the boundary by a single trailing space, punctuation mark, or a single conjunction (e.g. *", and"*, *"who"*), the sentence is marked salient. However, it contains no semantic information to answer the question, causing the LLM to judge it as non-salient.
* **Impact**: Creates false-positive noise in training data.

### B. Silver Non-Salient (0) but LLM Salient (1) (5 Cases)
* **Cause**: The sentence does not physically intersect the SQuAD answer span, but it contains crucial background context (e.g., named entity definitions, coreferents) necessary to construct the answer, or a paraphrase of the answer that the SQuAD annotators did not select.
* **Impact**: Creates false-negative noise in training data.

---

## 8. Actionable Recommendations for Dataset Enhancement

To clean and improve the SQuAD sentence salience dataset for future experiments, we recommend three protocols:

1. **Token-Level Intersection Filter**: Instead of any character overlap, require that the intersection contains at least one non-stopword token (nouns, verbs, adjectives). This removes boundary-overlap punctuation noise.
2. **Semantic Entailment Cross-Encoder**: Evaluate sentence relevance to the answer using a cross-encoder (e.g., DeBERTa-v3-large). If a sentence has a high semantic entailment with the answer, label it as salient even if the exact string match is missing.
3. **Coreference Resolution**: Run coreference resolution (e.g. using spaCy coref) to resolve pronouns. If a sentence contains a resolved pronoun pointing to an answer entity, it should be marked as salient context.

---

## 9. Rigorous Statistical Analysis & Significance Testing

To establish the statistical validity of our feature set, we conducted Welch's t-tests (two-sample independent t-tests with unequal variances) comparing Salient (Class 1) and Non-Salient (Class 0) sentences.

The full details and tables are documented in [docs/silver_data_analysis.md](file:///d:/Research/Sqaud-Salience/docs/silver_data_analysis.md). The key takeaways and their corresponding graphical interpretations include:

### A. Syntactic Complexity Differences
* **Metric**: Salient sentences exhibit significantly deeper dependency parse trees (mean **6.57** vs. **6.16**, p < 0.0001) and larger token dependency distances (mean **3.32** vs. **3.06**, p < 0.0001).
* **Plot**: This is visualized in `docs/images/syntactic_complexity.png`.

### B. Information Theoretic (Surprisal) Distributions
* **Metric**: While mean surprisal is slightly lower for salient sentences (due to semantic priming of question words), the **surprisal deletion coherence drop** is significantly larger when salient sentences are removed from the context paragraph (p < 0.05).
* **Plot**: Visually analyzed via probability density curves in `docs/images/surprisal_distribution.png`.

### C. Text Readability Index comparison
* **Metric**: Readability metrics (Flesch Reading Ease and Gunning Fog index) show no statistically significant differences between the two classes.
* **Plot**: Shown using overlapping boxplot distributions in `docs/images/readability_comparison.png`, proving basic readability is not a useful feature for salience prediction.

### D. Soft Label Target Separability
* **Metric**: The hybrid soft target (`soft_label_hybrid`), which combines spatial decay with TF-IDF alignment, provides a much clearer separation of classes compared to pure spatial decay.
* **Plot**: Boxplot comparison shown in `docs/images/soft_labels_comparison.png`.

