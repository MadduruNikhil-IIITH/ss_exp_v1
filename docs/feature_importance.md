# SQuAD Sentence Salience - Feature Importance Analysis

This document presents the standardized coefficients across our 4 Logistic Regression subsystems under **Discourse-Semantic Neighborhood Balancing (DSNB)** on the new 90/10 split.

> [!NOTE]
> All tabular features were scaled to zero mean and unit variance before fitting. Therefore, the absolute value of the coefficient directly represents the feature's relative importance (effect size).

---

## 1. Feature Coefficients Carousel (DSNB Balancing)

Use the carousel below to browse the standardized coefficients for each configuration. We display the top positive and negative features.

````carousel
### 1. Combined Model (Config 5)
*Standardized coefficients for the combined model integrating all engineered discourse, surprisal, and linguistic features under DSNB balancing.*

| Feature Name | Coefficient | Direction | Type | Description |
| :--- | :---: | :---: | :---: | :--- |
| `flesch_kincaid_grade` | `+0.6694` | Positive (+) | Linguistic / Readability | Flesch-Kincaid school grade level |
| `sentiment_polarity_compound` | `+0.5171` | Positive (+) | Linguistic / Sentiment | VADER compound sentiment polarity |
| `rel_surp_causal_pf_diff` | `-0.4619` | Negative (-) | Surprisal (GPT-2) | Surprisal difference relative to passage mean |
| `surp_causal_pf_sum` | `-0.4473` | Negative (-) | Surprisal (GPT-2) | Sum of causal word surprisals in sentence |
| `rel_rst_depth_ratio` | `-0.4352` | Negative (-) | Discourse (RST) | Mean sentence depth divided by max document tree depth |
| `rel_surp_pll_pf_ratio` | `+0.3959` | Positive (+) | Surprisal (BERT) | PLL surprisals ratio relative to passage mean |
| `rel_surp_causal_pf_sum_ratio` | `+0.3915` | Positive (+) | Surprisal (GPT-2) | Causal surprisal sum ratio relative to passage |
| `rel_surp_pll_pf_diff` | `-0.3858` | Negative (-) | Surprisal (BERT) | PLL surprisals difference relative to passage mean |
| `char_count` | `+0.3604` | Positive (+) | Linguistic / Syntax | Count of characters in sentence |
| `psg_rst_max_depth` | `-0.3583` | Negative (-) | Discourse (RST) | Max RST depth of document tree |

<!-- slide -->
### 2. Surprisal-Only Subsystem (Config 4)
*Standardized coefficients for the subsystem model trained exclusively on GPT-2 and BERT surprisal features under DSNB balancing.*

| Feature Name | Coefficient | Direction | Type | Description |
| :--- | :---: | :---: | :---: | :--- |
| `rel_surp_causal_pf_sum_ratio` | `+0.7189` | Positive (+) | Surprisal (GPT-2) | Causal surprisal sum ratio relative to passage |
| `rel_surp_pll_pf_sum_ratio` | `-0.5168` | Negative (-) | Surprisal (BERT) | PLL surprisal sum ratio relative to passage |
| `surp_pll_pf_sum` | `+0.4409` | Positive (+) | Surprisal (BERT) | Sum of PLL surprisals in sentence |
| `rel_surp_causal_pf_diff` | `-0.4191` | Negative (-) | Surprisal (GPT-2) | Surprisal difference relative to passage mean |
| `rel_surp_causal_pf_ratio` | `+0.3326` | Positive (+) | Surprisal (GPT-2) | Causal surprisal ratio relative to passage mean |
| `rel_surp_pll_pf_ratio` | `+0.3038` | Positive (+) | Surprisal (BERT) | PLL surprisals ratio relative to passage mean |
| `rel_surp_pll_pf_diff` | `-0.2682` | Negative (-) | Surprisal (BERT) | PLL surprisals difference relative to passage mean |
| `surp_causal_pf_max` | `-0.2545` | Negative (-) | Surprisal (GPT-2) | Max word surprisal value in sentence |
| `surp_causal_pf_std` | `+0.2153` | Positive (+) | Surprisal (GPT-2) | Standard deviation of causal word surprisals |
| `surp_deletion_drop` | `-0.2129` | Negative (-) | Surprisal (GPT-2) | Information drop when deleting target sentence |

<!-- slide -->
### 3. Linguistic-Only Subsystem (Config 3)
*Standardized coefficients for the subsystem model trained exclusively on linguistic, syntactic, and readability features under DSNB balancing.*

| Feature Name | Coefficient | Direction | Type | Description |
| :--- | :---: | :---: | :---: | :--- |
| `flesch_kincaid_grade` | `+0.6557` | Positive (+) | Readability | Flesch-Kincaid school grade level |
| `sentiment_polarity_compound` | `+0.4415` | Positive (+) | Sentiment | VADER compound sentiment polarity |
| `sentiment_polarity_pos` | `-0.2926` | Negative (-) | Sentiment | Positive sentiment score ratio |
| `sentiment_polarity_neg` | `+0.2668` | Positive (+) | Sentiment | Negative sentiment score ratio |
| `discourse_contrast_count` | `-0.2531` | Negative (-) | Sentiment / Connectives | Count of contrastive connectives |
| `flesch_reading_ease` | `+0.2486` | Positive (+) | Readability | Flesch Reading Ease index score |
| `number_ratio` | `+0.2429` | Positive (+) | Lexical | Ratio of numbers/facts in sentence |
| `pron_1st_ratio` | `-0.2344` | Negative (-) | Lexical | Ratio of first-person pronouns |
| `word_count` | `+0.2084` | Positive (+) | Length / Syntax | Count of words in sentence |
| `colon_count` | `+0.2054` | Positive (+) | Length / Syntax | Count of colons |

<!-- slide -->
### 4. Discourse-Only Subsystem (Config 2)
*Standardized coefficients for the subsystem model trained exclusively on RST discourse features under DSNB balancing.*

| Feature Name | Coefficient | Direction | Type | Description |
| :--- | :---: | :---: | :---: | :--- |
| `rel_rst_depth_ratio` | `-0.3065` | Negative (-) | Discourse (RST) | Mean sentence depth divided by max document tree depth |
| `rst_is_root` | `+0.2853` | Positive (+) | Discourse (RST) | Boolean indicating if sentence contains document root EDU |
| `rel_rst_n_ratio` | `+0.2289` | Positive (+) | Discourse (RST) | Ratio of sentence nuclei to document total nuclei |
| `psg_rst_max_depth` | `-0.2031` | Negative (-) | Discourse (RST) | Max RST depth of document tree |
| `psg_rst_n_count` | `+0.1926` | Positive (+) | Discourse (RST) | Total nuclei in document tree |
| `rst_s_count` | `+0.1659` | Positive (+) | Discourse (RST) | Count of satellites in sentence RST subtree |
| `rst_edu_count` | `+0.1272` | Positive (+) | Discourse (RST) | Count of Elementary Discourse Units (EDUs) in sentence |
| `rst_rel_elaboration_count` | `+0.1022` | Positive (+) | Discourse (RST) | Count of elaboration RST relations |
| `rst_n_count` | `+0.0608` | Positive (+) | Discourse (RST) | Count of nuclei in sentence RST subtree |
| `psg_rst_s_count` | `-0.0585` | Negative (-) | Discourse (RST) | Total satellites count in document tree |
````

---

## 2. Model & Dataset Configuration Details

*   **Balancing Configuration**: Discourse-Semantic Neighborhood Balancing (DSNB)
*   **Training Dataset Size**: `892` records (`446` salient and `446` non-salient sentences)
*   **Preprocessing**: All features standardized (scaled to zero mean and unit variance) prior to fitting.
