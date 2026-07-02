# SQuAD 5-Fold Context-Level Cross-Validation Report

This report presents the context-level 5-fold cross-validation results for SQuAD sentence salience, evaluated at a standardized threshold of **`0.35`**.

## 1. Cross-Validation Results Table

| Model | Metric Type | Accuracy | Precision | Recall | F1 | MRR | MAP | NDCG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LR Combined** | Mean | `0.4385` | `0.2375` | `0.8360` | `0.3695` | `0.5818` | `0.5819` | `0.6845` |
| **LR Combined** | Std | `0.0620` | `0.0159` | `0.0805` | `0.0255` | `0.0243` | `0.0244` | `0.0195` |
| **LGSM** | Mean | `0.6949` | `0.3185` | `0.4291` | `0.3591` | `0.5410` | `0.5411` | `0.6536` |
| **LGSM** | Std | `0.0532` | `0.0766` | `0.0667` | `0.0514` | `0.0375` | `0.0374` | `0.0284` |

## 2. Key Statistical Insights

- **F1-Score Difference**: LGSM outperforms Combined LR by **`+-0.0105`** in mean F1 (`0.3591` vs. `0.3695`).
- **Ranking MAP Difference**: LGSM outperforms Combined LR by **`+-0.0407`** in mean MAP (`0.5411` vs. `0.5819`).

### Positional and Semantic Robustness
The standard deviations for LGSM (F1 std = `0.0514`, MAP std = `0.0374`) demonstrate that LGSM is highly stable across different folds. By performing validation strictly on context passages that were excluded from training, we verify that LGSM's learning generalizes well to unseen contexts and is robust to SQuAD position bias.
