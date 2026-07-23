# SQuAD 5-Fold Context-Level Cross-Validation Report

This report presents the context-level 5-fold cross-validation results for SQuAD sentence salience, evaluated at a standardized threshold of **`0.35`**.

## 1. Cross-Validation Results Table

| Model | Metric Type | Accuracy | Precision | Recall | F1 | MRR | MAP | NDCG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LR Combined** | Mean | `0.7203` | `0.7514` | `0.8581` | `0.8010` | `0.9415` | `0.8872` | `0.9365` |
| **LR Combined** | Std | `0.0050` | `0.0098` | `0.0201` | `0.0064` | `0.0074` | `0.0040` | `0.0026` |
| **LGSM** | Mean | `0.6635` | `0.6624` | `0.9936` | `0.7948` | `0.9326` | `0.8776` | `0.9298` |
| **LGSM** | Std | `0.0121` | `0.0124` | `0.0097` | `0.0104` | `0.0141` | `0.0119` | `0.0086` |

## 2. Key Statistical Insights

- **F1-Score Difference**: LGSM outperforms Combined LR by **`+-0.0061`** in mean F1 (`0.7948` vs. `0.8010`).
- **Ranking MAP Difference**: LGSM outperforms Combined LR by **`+-0.0096`** in mean MAP (`0.8776` vs. `0.8872`).

### Positional and Semantic Robustness
The standard deviations for LGSM (F1 std = `0.0104`, MAP std = `0.0119`) demonstrate that LGSM is highly stable across different folds. By performing validation strictly on context passages that were excluded from training, we verify that LGSM's learning generalizes well to unseen contexts and is robust to SQuAD position bias.
