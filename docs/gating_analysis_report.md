# SQuAD LGSM Gating Behavior Analysis Report

This report analyzes the behavior of the scalar gating parameter $\alpha_t \in (0, 1)$ in the Linguistically-Grounded Saliency Model (LGSM) on SQuAD validation passages.

## 1. Gating Statistics Summary
- **Mean Gate Value**: `0.3369`
- **Median Gate Value**: `0.3379`
- **Standard Deviation**: `0.0102`
- **Linear Correlation ($r$)**: `0.2046` (p-value = `1.8532e-03`)

### Position-Wise Gate Values
| Sentence Index | Mean Gate Value (alpha) | Std Dev | Sample Count |
| --- | --- | --- | --- |
| Sentence 0 | `0.3377` | `0.0066` | 39 |
| Sentence 1 | `0.3357` | `0.0090` | 39 |
| Sentence 2 | `0.3364` | `0.0077` | 33 |
| Sentence 3 | `0.3339` | `0.0076` | 33 |
| Sentence 4 | `0.3405` | `0.0096` | 27 |
| Sentence 5 | `0.3346` | `0.0082` | 23 |
| Sentence 6 | `0.3322` | `0.0022` | 18 |
| Sentence 7 | `0.3554` | `0.0194` | 13 |

## 2. Key Findings & Discussion

### A. Stream Weight Distribution
The mean gate value of `0.3369` indicates that the model relies **more heavily on the semantic stream** (BERT representations) than on explicit features.

### B. Positional Gating Progression (Arc)
The gate value progresses with a slope of `0.0022` across sentence indices. This flat temporal trend indicates that linguistic features are continuously and evenly integrated throughout the passage, rather than isolated to specific positions.

### C. Gating-Prediction Correlation (Negative Filter Hypothesis)
The low correlation of `0.2046` suggests that the gating coefficient adjusts dynamically and relationally to select salient sentences, rather than acting as a simple monotonic filter.
