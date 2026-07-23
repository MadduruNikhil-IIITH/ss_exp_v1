# SQuAD Token-Level Surprisal UID Analysis Report

This report analyzes token-level information density in SQuAD context sentences to test the **Uniform Information Density (UID)** hypothesis, investigating whether salient sentences suppress rare, hard-to-predict words.

## 1. Quantile Comparison Table
| Quantile | Salient (bits) | Non-Salient (bits) | Difference (Salient - Non) |
| --- | --- | --- | --- |
| P5 | `0.0537` | `0.0386` | **`0.0151`** |
| P25 | `1.4258` | `1.2645` | **`0.1613`** |
| P50 | `4.0921` | `3.8807` | **`0.2114`** |
| P75 | `8.0154` | `8.0054` | **`0.0101`** |
| P90 | `11.8109` | `11.8440` | **`-0.0331`** |
| P95 | `14.2056` | `14.6192` | **`-0.4136`** |
| P99 | `18.9432` | `18.8976` | **`0.0456`** |

### Statistical Test
- **Kolmogorov-Smirnov Test**: KS-statistic = `0.0203` ($p = 1.6462e-02$)

## 2. Key Findings & Discussion

- **Median Difference (P50)**: `0.2114` bits.
- **Upper-Tail Difference (P90)**: `-0.0331` bits.
- **Upper-Tail Difference (P95)**: `-0.4136` bits.

### Upper-Tail Surprisal Asymmetry
The results do not show a pronounced upper-tail suppression in salient sentences. The distributions remain close across both quantiles.
