# SQuAD Silver Data Statistical Analysis

This document provides a rigorous statistical analysis of the SQuAD sentence-level silver datasets used in our salience experiments.

---

## 1. Dataset Dimensions and Splits

| Metric | Training Set | Validation Set | Total |
| :--- | :---: | :---: | :---: |
| **Unique Contexts** | 60 | 15 | 75 |
| **QA Pairs (Questions)** | 337 | 303 | 640 |
| **Sentence-Question Records** | 2156 | 1322 | 3478 |
| **Average Sentences per Context** | 6.40 (Min: 3, Max: 13) | - | - |

---

## 2. Class Imbalance Profile

Since each question typically has exactly one sentence containing the answer span, the dataset is inherently imbalanced.

* **Training Set Class Distribution**:
  * **Salient (Class 1)**: 339 (15.72%)
  * **Non-Salient (Class 0)**: 1817 (84.28%)
  * **Imbalance Ratio**: ~1 : 5.4
* **Validation Set Class Distribution**:
  * **Salient (Class 1)**: 336 (25.42%)
  * **Non-Salient (Class 0)**: 986 (74.58%)
  * **Imbalance Ratio**: ~1 : 2.9

---

## 3. Positional Bias (Where do answers reside?)

The table below shows the distribution of Class 1 (salient answer sentences) by their linear index in the context passage.

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
> **Extreme Positional Bias**: Over **75%** of all salient sentences reside at Sentence Index 0, 1, or 2. This represents a significant shortcut that models can exploit (e.g., simply predicting that early sentences are salient). This highlights the critical importance of balancing methods like **DSNB** which mine negatives from the same positional neighborhoods to break this bias.

---

## 4. Feature Correlations with Salience

Below is the Pearson correlation coefficient ($r$) between salient labels (`binary_label`) and our extracted features in the training dataset.

| Feature Name | Pearson Correlation ($r$) | Category | Interpretation |
| :--- | :---: | :---: | :--- |
| `word_count` | +0.1336 | Linguistic / Length | Weak Positive correlation |
| `char_count` | +0.1154 | Linguistic / Length | Weak Positive correlation |
| `align_sem_sim` | +0.4925 | Semantic Alignment | Strong Positive correlation |
| `align_rouge_l_recall` | +0.4441 | Semantic Alignment | Strong Positive correlation |
| `align_jaccard` | +0.4706 | Semantic Alignment | Strong Positive correlation |
| `rel_rst_n_ratio` | +0.1654 | Discourse (RST) | Moderate Positive correlation |
| `rst_mean_depth` | -0.0674 | Discourse (RST) | Weak Negative correlation |
| `surp_mean` | +0.0117 | Surprisal (GPT-2) | Weak Positive correlation |
| `surp_std` | +0.0502 | Surprisal (GPT-2) | Weak Positive correlation |
| `rel_surp_ratio` | +0.0040 | Surprisal (GPT-2) | Weak Positive correlation |
| `surp_deletion_drop` | +0.0348 | Surprisal (GPT-2) | Weak Positive correlation |

---

## 5. Potential Improvements to Silver Data

Our LLM-as-a-Judge validation verified that exact boundary intersection labels have an 82% agreement with human-aligned LLM judgments, but highlighted two key limitations:
1. **Paraphrase Missing (False Negatives)**: exact overlap fails to label sentences that contain paraphrased or coreferent mentions of the answer.
2. **Boundary Overlap Noise (False Positives)**: sentences containing only a trailing space or a single punctuation mark of the answer span are labeled as Class 1.

### Recommended Data Cleaning and Enhancement Protocol:
* **Token-Level Intersection Filter**: Label a sentence as Class 1 only if the intersection contains at least one non-stopword token of the answer, preventing punctuation-only overlap.
* **Coreference Resolution**: Run coreference resolution (e.g., using spaCy's coref resolver) to link pronouns (like *he*, *she*, *they*, *it*) in context sentences to the named entities in the question/answer, mapping salient contexts more accurately.
* **Semantic Coverage Thresholding**: Use a cross-encoder to compute sentence-answer similarity, labeling a sentence as salient if it has a high entailment score with the answer context, even without exact word overlap.
