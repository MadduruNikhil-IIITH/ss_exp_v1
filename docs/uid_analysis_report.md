# SQuAD Token-Level Surprisal UID Analysis Report

This report analyzes token-level information density in SQuAD context sentences to test the **Uniform Information Density (UID)** hypothesis, investigating whether salient sentences suppress rare, hard-to-predict words.

## 1. Quantile Comparison Table
| Quantile | Salient (bits) | Non-Salient (bits) | Difference (Salient - Non) |
| --- | --- | --- | --- |
| P5 | `0.0599` | `0.0251` | **`0.0348`** |
| P25 | `1.4040` | `1.4037` | **`0.0004`** |
| P50 | `4.4052` | `4.1792` | **`0.2260`** |
| P75 | `8.5854` | `8.0945` | **`0.4909`** |
| P90 | `12.7368` | `11.4483` | **`1.2885`** |
| P95 | `15.4360` | `14.2236` | **`1.2124`** |
| P99 | `19.4828` | `18.9758` | **`0.5070`** |

### Statistical Test
- **Kolmogorov-Smirnov Test**: KS-statistic = `0.0468` ($p = 4.9334e-01$)

## 2. Key Findings & Discussion

- **Median Difference (P50)**: `0.2260` bits.
- **Upper-Tail Difference (P90)**: `1.2885` bits.
- **Upper-Tail Difference (P95)**: `1.2124` bits.

### Upper-Tail Surprisal Asymmetry
The results do not show a pronounced upper-tail suppression in salient sentences. The distributions remain close across both quantiles.
