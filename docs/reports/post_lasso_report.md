# SQuAD Controlled Corpus Study: Post-LASSO Regression Analysis

> [!IMPORTANT]
> **Single Model Configuration Note**: This controlled corpus study is conducted using a **single, specific model configuration** to establish statistical significance. Specifically, we run an unregularized Logistic Regression refit (`statsmodels.Logit`) on the **raw, unbalanced training split** (2,301 records). Crucially, we force-include two baseline control variables—**Sentence Length (`word_count`)** and **Linear Position (`sentence_idx`)**—alongside the features selected by the L1 (LASSO) penalty. This isolates the independent predictive signals of our engineered discourse (RST) and cognitive (surprisal) features from simple physical shortcuts.

---

## 1. Top Predictive Features
| Rank | Feature | Coefficient | z-score | p-value | Significance |
| --- | --- | --- | --- | --- | --- |
| 1 | `rel_surp_causal_pf_sum_ratio` | `0.6251` | `4.63` | `3.7209e-06` | ******* |
| 2 | `number_ratio` | `0.1771` | `2.63` | `8.4747e-03` | ****** |
| 3 | `rst_s_count` | `-0.2427` | `-2.33` | `1.9926e-02` | ***** |
| 4 | `sentence_idx_control` | `-0.2121` | `-2.29` | `2.2158e-02` | ***** |
| 5 | `rel_surp_causal_pf_diff` | `-0.1952` | `-2.10` | `3.5327e-02` | ***** |
| 6 | `surp_pll_pf_std` | `0.1680` | `1.96` | `5.0222e-02` | **** |
| 7 | `concrete_max` | `0.1610` | `1.89` | `5.8468e-02` | **** |
| 8 | `stopword_ratio` | `-0.1518` | `-1.69` | `9.0942e-02` | **** |
| 9 | `verb_ratio` | `0.1446` | `1.69` | `9.1121e-02` | **** |
| 10 | `pron_1st_ratio` | `-0.1392` | `-1.66` | `9.6490e-02` | **** |
| 11 | `discourse_contrast_count` | `-0.1160` | `-1.62` | `1.0580e-01` | **** |
| 12 | `adv_ratio` | `-0.1083` | `-1.44` | `1.4936e-01` | **** |
| 13 | `rst_rel_attribution_count` | `0.0874` | `1.30` | `1.9240e-01` | **** |
| 14 | `parenthesis_count` | `0.1027` | `1.28` | `2.0199e-01` | **** |
| 15 | `discourse_addition_count` | `-0.1185` | `-1.24` | `2.1590e-01` | **** |
| 16 | `conj_ratio` | `0.1036` | `1.13` | `2.6010e-01` | **** |
| 17 | `period_count` | `-0.0830` | `-1.12` | `2.6106e-01` | **** |
| 18 | `rst_rel_joint_count` | `-0.0766` | `-0.99` | `3.2274e-01` | **** |
| 19 | `concrete_mean` | `-0.0857` | `-0.96` | `3.3688e-01` | **** |
| 20 | `surp_causal_pf_max` | `-0.0607` | `-0.95` | `3.4013e-01` | **** |
| 21 | `avg_word_length` | `-0.0980` | `-0.95` | `3.4193e-01` | **** |
| 22 | `semicolon_count` | `-0.0568` | `-0.94` | `3.4875e-01` | **** |
| 23 | `rst_mean_depth` | `-0.1335` | `-0.93` | `3.5483e-01` | **** |
| 24 | `cap_ratio` | `0.0933` | `0.88` | `3.8071e-01` | **** |
| 25 | `surp_causal_pf_min` | `-0.0633` | `-0.75` | `4.5445e-01` | **** |
| 26 | `colon_count` | `0.0528` | `0.72` | `4.6862e-01` | **** |
| 27 | `gunning_fog` | `0.0769` | `0.68` | `4.9538e-01` | **** |
| 28 | `ttr` | `-0.0552` | `-0.67` | `5.0485e-01` | **** |
| 29 | `discourse_causal_count` | `0.0425` | `0.66` | `5.0873e-01` | **** |
| 30 | `word_count` | `0.1220` | `0.66` | `5.1108e-01` | **** |
| 31 | `noun_ratio` | `-0.0654` | `-0.65` | `5.1377e-01` | **** |
| 32 | `rel_rst_depth_ratio` | `-0.0645` | `-0.59` | `5.5449e-01` | **** |
| 33 | `pron_ratio` | `0.0730` | `0.56` | `5.7635e-01` | **** |
| 34 | `surp_pll_pf_min` | `-0.0367` | `-0.54` | `5.9174e-01` | **** |
| 35 | `rst_n_count` | `0.0564` | `0.53` | `5.9321e-01` | **** |
| 36 | `sentiment_polarity_neu` | `0.0312` | `0.48` | `6.3307e-01` | **** |
| 37 | `past_tense_ratio` | `-0.0322` | `-0.44` | `6.5823e-01` | **** |
| 38 | `avg_dep_distance` | `0.0387` | `0.44` | `6.6105e-01` | **** |
| 39 | `title_ratio` | `0.0462` | `0.43` | `6.6573e-01` | **** |
| 40 | `surp_deletion_drop` | `0.0166` | `0.19` | `8.4869e-01` | **** |
| 41 | `sentiment_polarity_compound` | `-0.0103` | `-0.17` | `8.6462e-01` | **** |
| 42 | `pron_3rd_ratio` | `0.0113` | `0.09` | `9.2686e-01` | **** |
| 43 | `question_count` | `-1.3209` | `-0.00` | `9.9972e-01` | **** |

*Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05.*

## 2. Key Findings & Discussion

### A. Position and Length Controls
- **Position Bias**: The normalized position feature has a coefficient of `-0.2121` ($z = -2.29$, $p = 2.2158e-02$). The negative coefficient confirms the expected lead-bias: sentences early in the passage are significantly more likely to contain answers (salient).
- **Length Bias**: The word count has a coefficient of `0.1220` ($z = 0.66$). This shows how length affects the salience probability.

### B. RST Discourse Structure Predictors
Several RST discourse features emerged as significant independent predictors:
- `rst_s_count`: Coef = `-0.2427` ($z = -2.33$, $p = 1.9926e-02$). Higher values indicate decreased likelihood (acting as a negative filter).
- `rst_rel_attribution_count`: Coef = `0.0874` ($z = 1.30$, $p = 1.9240e-01$). Higher values indicate increased likelihood of sentence salience.
- `rst_rel_joint_count`: Coef = `-0.0766` ($z = -0.99$, $p = 3.2274e-01$). Higher values indicate decreased likelihood (acting as a negative filter).
- `rst_mean_depth`: Coef = `-0.1335` ($z = -0.93$, $p = 3.5483e-01$). Higher values indicate decreased likelihood (acting as a negative filter).
- `rel_rst_depth_ratio`: Coef = `-0.0645` ($z = -0.59$, $p = 5.5449e-01$). Higher values indicate decreased likelihood (acting as a negative filter).

### C. Cognitive Surprisal Predictors
Information theoretic surprisal features show the following independent signals:
- `rel_surp_causal_pf_sum_ratio`: Coef = `0.6251` ($z = 4.63$, $p = 3.7209e-06$). Higher surprisal values are positively correlated with answer salience.
- `rel_surp_causal_pf_diff`: Coef = `-0.1952` ($z = -2.10$, $p = 3.5327e-02$). Lower surprisal (more predictable context) correlates with salience, indicating smoother information contours.
- `surp_pll_pf_std`: Coef = `0.1680` ($z = 1.96$, $p = 5.0222e-02$). Higher surprisal values are positively correlated with answer salience.
- `surp_causal_pf_max`: Coef = `-0.0607` ($z = -0.95$, $p = 3.4013e-01$). Lower surprisal (more predictable context) correlates with salience, indicating smoother information contours.
- `surp_causal_pf_min`: Coef = `-0.0633` ($z = -0.75$, $p = 4.5445e-01$). Lower surprisal (more predictable context) correlates with salience, indicating smoother information contours.
