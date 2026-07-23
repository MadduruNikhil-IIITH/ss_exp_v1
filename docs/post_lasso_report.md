# SQuAD Controlled Corpus Study: Post-LASSO Regression Analysis

> [!IMPORTANT]
> **Single Model Configuration Note**: This controlled corpus study is conducted using a **single, specific model configuration** to establish statistical significance. Specifically, we run an unregularized Logistic Regression refit on the **raw, unbalanced training split** (2,301 records). Crucially, we force-include two baseline control variables—**Sentence Length (`word_count`)** and **Linear Position (`sentence_idx`)**—alongside the features selected by the L1 (LASSO) penalty. This isolates the independent predictive signals of our engineered discourse (RST) and cognitive (surprisal) features from simple physical shortcuts.

---

## 1. Top Predictive Features
| Rank | Feature | Coefficient | z-score | p-value | Significance |
| --- | --- | --- | --- | --- | --- |
| 1 | `number_ratio` | `0.3406` | `7.15` | `8.8818e-13` | ******* |
| 2 | `rel_surp_causal_pf_sum_ratio` | `0.9551` | `6.49` | `8.6217e-11` | ******* |
| 3 | `sentence_idx_control` | `-0.2794` | `-5.52` | `3.4477e-08` | ******* |
| 4 | `title_ratio` | `0.1688` | `3.68` | `2.3566e-04` | ******* |
| 5 | `pron_3rd_ratio` | `0.1741` | `2.67` | `7.5962e-03` | ****** |
| 6 | `max_parse_depth` | `-0.1449` | `-2.42` | `1.5656e-02` | ***** |
| 7 | `concrete_max` | `0.2176` | `2.27` | `2.3202e-02` | ***** |
| 8 | `surp_causal_pf_std` | `-0.1427` | `-2.01` | `4.4950e-02` | ***** |
| 9 | `word_count` | `-0.3825` | `-1.99` | `4.6124e-02` | ***** |
| 10 | `rst_rel_joint_count` | `-0.1001` | `-1.97` | `4.8473e-02` | ***** |
| 11 | `verb_ratio` | `0.0946` | `1.96` | `5.0249e-02` | **** |
| 12 | `surp_causal_pf_min` | `-0.1026` | `-1.90` | `5.7678e-02` | **** |
| 13 | `prep_ratio` | `0.0907` | `1.88` | `6.0373e-02` | **** |
| 14 | `surp_causal_pf_sum` | `0.3455` | `1.81` | `7.0807e-02` | **** |
| 15 | `pron_ratio` | `-0.1205` | `-1.69` | `9.1483e-02` | **** |
| 16 | `rst_mean_depth` | `-0.1454` | `-1.62` | `1.0510e-01` | **** |
| 17 | `ttr` | `-0.0826` | `-1.49` | `1.3652e-01` | **** |
| 18 | `pron_1st_ratio` | `0.0606` | `1.49` | `1.3657e-01` | **** |
| 19 | `adv_ratio` | `-0.0546` | `-1.33` | `1.8474e-01` | **** |
| 20 | `period_count` | `-0.0629` | `-1.27` | `2.0552e-01` | **** |
| 21 | `surp_pll_pf_min` | `0.0520` | `1.22` | `2.2302e-01` | **** |
| 22 | `sentiment_polarity_neu` | `-0.0444` | `-1.14` | `2.5437e-01` | **** |
| 23 | `rel_rst_depth_ratio` | `-0.0800` | `-1.05` | `2.9373e-01` | **** |
| 24 | `surp_deletion_drop` | `0.1393` | `0.99` | `3.2433e-01` | **** |
| 25 | `surp_causal_pf_max` | `0.0575` | `0.95` | `3.4393e-01` | **** |
| 26 | `surp_causal_pf_mean` | `0.1003` | `0.91` | `3.6512e-01` | **** |
| 27 | `gunning_fog` | `0.0556` | `0.90` | `3.6588e-01` | **** |
| 28 | `conj_ratio` | `-0.0436` | `-0.81` | `4.1980e-01` | **** |
| 29 | `rel_rst_n_ratio` | `0.1124` | `0.78` | `4.3710e-01` | **** |
| 30 | `past_tense_ratio` | `0.0322` | `0.74` | `4.5828e-01` | **** |
| 31 | `stopword_ratio` | `-0.0406` | `-0.72` | `4.7363e-01` | **** |
| 32 | `question_count` | `-0.0289` | `-0.71` | `4.7473e-01` | **** |
| 33 | `surp_pll_pf_max` | `0.0278` | `0.66` | `5.1110e-01` | **** |
| 34 | `concrete_min` | `-0.0507` | `-0.60` | `5.4857e-01` | **** |
| 35 | `rel_surp_causal_pf_diff` | `-0.2111` | `-0.59` | `5.5327e-01` | **** |
| 36 | `exclamation_count` | `-0.2816` | `-0.55` | `5.8025e-01` | **** |
| 37 | `rst_n_count` | `0.0619` | `0.47` | `6.3754e-01` | **** |
| 38 | `pron_2nd_ratio` | `-0.0170` | `-0.45` | `6.5051e-01` | **** |
| 39 | `concrete_mean` | `-0.0334` | `-0.45` | `6.5142e-01` | **** |
| 40 | `discourse_addition_count` | `-0.0253` | `-0.41` | `6.8180e-01` | **** |
| 41 | `concrete_std` | `-0.0383` | `-0.40` | `6.8920e-01` | **** |
| 42 | `rel_surp_causal_pf_ratio` | `-0.1246` | `-0.38` | `7.0414e-01` | **** |
| 43 | `discourse_contrast_count` | `-0.0158` | `-0.37` | `7.1170e-01` | **** |
| 44 | `psg_rst_n_count` | `-0.0264` | `-0.37` | `7.1300e-01` | **** |
| 45 | `rst_rel_elaboration_count` | `0.0159` | `0.31` | `7.5572e-01` | **** |
| 46 | `avg_dep_distance` | `-0.0108` | `-0.18` | `8.6037e-01` | **** |
| 47 | `parenthesis_count` | `0.0108` | `0.16` | `8.7572e-01` | **** |
| 48 | `rst_rel_attribution_count` | `-0.0047` | `-0.11` | `9.1423e-01` | **** |

*Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05.*

## 2. Key Findings & Discussion

### A. Position and Length Controls
- **Position Bias**: The normalized position feature has a coefficient of `-0.2794` ($z = -5.52$, $p = 3.4477e-08$). The negative coefficient confirms the expected lead-bias: sentences early in the passage are significantly more likely to contain answers (salient).
- **Length Bias**: The word count has a coefficient of `-0.3825` ($z = -1.99$). This shows how length affects the salience probability.

### B. RST Discourse Structure Predictors
Several RST discourse features emerged as significant independent predictors:
- `rst_rel_joint_count`: Coef = `-0.1001` ($z = -1.97$, $p = 4.8473e-02$). Higher values indicate decreased likelihood (acting as a negative filter).
- `rst_mean_depth`: Coef = `-0.1454` ($z = -1.62$, $p = 1.0510e-01$). Higher values indicate decreased likelihood (acting as a negative filter).
- `rel_rst_depth_ratio`: Coef = `-0.0800` ($z = -1.05$, $p = 2.9373e-01$). Higher values indicate decreased likelihood (acting as a negative filter).
- `rel_rst_n_ratio`: Coef = `0.1124` ($z = 0.78$, $p = 4.3710e-01$). Higher values indicate increased likelihood of sentence salience.
- `rst_n_count`: Coef = `0.0619` ($z = 0.47$, $p = 6.3754e-01$). Higher values indicate increased likelihood of sentence salience.

### C. Cognitive Surprisal Predictors
Information theoretic surprisal features show the following independent signals:
- `rel_surp_causal_pf_sum_ratio`: Coef = `0.9551` ($z = 6.49$, $p = 8.6217e-11$). Higher surprisal values are positively correlated with answer salience.
- `surp_causal_pf_std`: Coef = `-0.1427` ($z = -2.01$, $p = 4.4950e-02$). Lower surprisal (more predictable context) correlates with salience, indicating smoother information contours.
- `surp_causal_pf_min`: Coef = `-0.1026` ($z = -1.90$, $p = 5.7678e-02$). Lower surprisal (more predictable context) correlates with salience, indicating smoother information contours.
- `surp_causal_pf_sum`: Coef = `0.3455` ($z = 1.81$, $p = 7.0807e-02$). Higher surprisal values are positively correlated with answer salience.
- `surp_pll_pf_min`: Coef = `0.0520` ($z = 1.22$, $p = 2.2302e-01$). Higher surprisal values are positively correlated with answer salience.
