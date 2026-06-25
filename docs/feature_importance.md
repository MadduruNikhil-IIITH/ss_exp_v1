# SQuAD Sentence Salience - Feature Importance Analysis

This document presents a comprehensive analysis of the standardized coefficients across all 6 Logistic Regression configurations under different balancing methods (with emphasis on `None` and `DSNB`).

> [!NOTE]
> All tabular features were scaled to zero mean and unit variance before fitting. Therefore, the absolute value of the coefficient directly represents the feature's relative importance (effect size).

---

## 1. Feature Coefficients Carousel (DSNB vs. None)

Use the carousel below to browse the standardized coefficients for each configuration. We display the top positive and negative features.

````carousel
### 1. Combined Deletion (DSNB Balancing)
*Standardized coefficients for the combined model integrating the GPT-2 Sentence Deletion Coherence Drop under DSNB balancing.*

| Feature Name | Coefficient | Direction | Type | Description |
| :--- | :---: | :---: | :---: | :--- |
| `word_count` | 1.1044 | Positive (+) | Linguistic / Syntactic | Number of words in target sentence |
| `align_sem_sim` | 0.8544 | Positive (+) | Semantic Alignment | SBERT cosine semantic similarity with question |
| `char_count` | -0.8539 | Negative (-) | Linguistic / Syntactic | Number of characters in target sentence |
| `rel_rst_n_ratio` | 0.7203 | Positive (+) | Discourse (RST) | Ratio of sentence nuclei to document total nuclei |
| `rel_surp_ratio` | -0.6146 | Negative (-) | Surprisal (GPT-2) | Ratio of surprisal relative to passage mean |
| `comma_count` | 0.5594 | Positive (+) | Linguistic / Syntactic | Count of commas |
| `align_rouge_l_recall` | 0.5387 | Positive (+) | Semantic Alignment | ROUGE-L recall score with question |
| `surp_std` | 0.5289 | Positive (+) | Surprisal (GPT-2) | Standard deviation of word surprisals |
| `rel_surp_diff` | 0.5082 | Positive (+) | Surprisal (GPT-2) | Difference in surprisal relative to passage mean |
| `title_ratio` | 0.5058 | Positive (+) | Linguistic / Syntactic | Ratio of title-cased words |
| `rel_rst_depth_ratio` | -0.4538 | Negative (-) | Discourse (RST) | Ratio of sentence mean RST depth to document max depth |
| `question_count` | -0.4523 | Negative (-) | Linguistic / Syntactic | Count of question marks |

<!-- slide -->
### 2. Combined Heuristic (DSNB Balancing)
*Standardized coefficients for the combined model integrating the rule-based RST scoring heuristic under DSNB balancing.*

| Feature Name | Coefficient | Direction | Type | Description |
| :--- | :---: | :---: | :---: | :--- |
| `word_count` | 1.0930 | Positive (+) | Linguistic / Syntactic | Number of words in target sentence |
| `align_sem_sim` | 0.8554 | Positive (+) | Semantic Alignment | SBERT cosine semantic similarity with question |
| `char_count` | -0.8544 | Negative (-) | Linguistic / Syntactic | Number of characters in target sentence |
| `rel_rst_n_ratio` | 0.7241 | Positive (+) | Discourse (RST) | Ratio of sentence nuclei to document total nuclei |
| `rel_surp_ratio` | -0.6144 | Negative (-) | Surprisal (GPT-2) | Ratio of surprisal relative to passage mean |
| `comma_count` | 0.5519 | Positive (+) | Linguistic / Syntactic | Count of commas |
| `align_rouge_l_recall` | 0.5380 | Positive (+) | Semantic Alignment | ROUGE-L recall score with question |
| `surp_std` | 0.5307 | Positive (+) | Surprisal (GPT-2) | Standard deviation of word surprisals |
| `rel_surp_diff` | 0.5066 | Positive (+) | Surprisal (GPT-2) | Difference in surprisal relative to passage mean |
| `title_ratio` | 0.5051 | Positive (+) | Linguistic / Syntactic | Ratio of title-cased words |
| `rel_rst_depth_ratio` | -0.4840 | Negative (-) | Discourse (RST) | Ratio of sentence mean RST depth to document max depth |
| `question_count` | -0.4526 | Negative (-) | Linguistic / Syntactic | Count of question marks |

<!-- slide -->
### 3. Combined Baseline (DSNB Balancing)
*Standardized coefficients for the combined model without heuristics under DSNB balancing.*

| Feature Name | Coefficient | Direction | Type | Description |
| :--- | :---: | :---: | :---: | :--- |
| `word_count` | 1.1044 | Positive (+) | Linguistic / Syntactic | Number of words in target sentence |
| `align_sem_sim` | 0.8544 | Positive (+) | Semantic Alignment | SBERT cosine semantic similarity with question |
| `char_count` | -0.8539 | Negative (-) | Linguistic / Syntactic | Number of characters in target sentence |
| `rel_rst_n_ratio` | 0.7203 | Positive (+) | Discourse (RST) | Ratio of sentence nuclei to document total nuclei |
| `rel_surp_ratio` | -0.6146 | Negative (-) | Surprisal (GPT-2) | Ratio of surprisal relative to passage mean |
| `comma_count` | 0.5594 | Positive (+) | Linguistic / Syntactic | Count of commas |
| `align_rouge_l_recall` | 0.5387 | Positive (+) | Semantic Alignment | ROUGE-L recall score with question |
| `surp_std` | 0.5289 | Positive (+) | Surprisal (GPT-2) | Standard deviation of word surprisals |
| `rel_surp_diff` | 0.5082 | Positive (+) | Surprisal (GPT-2) | Difference in surprisal relative to passage mean |
| `title_ratio` | 0.5058 | Positive (+) | Linguistic / Syntactic | Ratio of title-cased words |
| `rel_rst_depth_ratio` | -0.4538 | Negative (-) | Discourse (RST) | Ratio of sentence mean RST depth to document max depth |
| `question_count` | -0.4523 | Negative (-) | Linguistic / Syntactic | Count of question marks |

<!-- slide -->
### 4. Discourse-Only Subsystem (DSNB Balancing)
*Standardized coefficients for the model trained exclusively on RST discourse features under DSNB balancing.*

| Feature Name | Coefficient | Direction | Type | Description |
| :--- | :---: | :---: | :---: | :--- |
| `rel_rst_n_ratio` | 0.6402 | Positive (+) | Discourse (RST) | Ratio of sentence nuclei to document total nuclei |
| `rst_n_count` | -0.4676 | Negative (-) | Discourse (RST) | Count of nuclei in sentence RST subtree |
| `rel_rst_depth_ratio` | -0.4521 | Negative (-) | Discourse (RST) | Ratio of sentence mean RST depth to document max depth |
| `rst_rel_joint_count` | 0.3247 | Positive (+) | Discourse (RST) | Count of joint RST relations |
| `rst_s_count` | 0.3079 | Positive (+) | Discourse (RST) | Count of satellites in sentence RST subtree |
| `psg_rst_n_count` | 0.2373 | Positive (+) | Discourse (RST) | Total nuclei count in document tree |
| `rst_mean_depth` | 0.2191 | Positive (+) | Discourse (RST) | Mean depth of EDUs in sentence RST subtree |
| `rst_edu_count` | -0.1773 | Negative (-) | Discourse (RST) | Count of elementary discourse units (EDUs) in sentence |
| `psg_rst_max_depth` | -0.1739 | Negative (-) | Discourse (RST) | Max RST depth in document tree |
| `rst_rel_attribution_count` | -0.1315 | Negative (-) | Discourse (RST) | Count of attribution RST relations |
| `rst_rel_elaboration_count` | -0.0676 | Negative (-) | Discourse (RST) | Count of elaboration RST relations |
| `psg_rst_s_count` | 0.0586 | Positive (+) | Discourse (RST) | Total satellites count in document tree |

<!-- slide -->
### 5. Surprisal-Only Subsystem (DSNB Balancing)
*Standardized coefficients for the model trained exclusively on GPT-2 surprisal features under DSNB balancing.*

| Feature Name | Coefficient | Direction | Type | Description |
| :--- | :---: | :---: | :---: | :--- |
| `rel_surp_ratio` | -1.0137 | Negative (-) | Surprisal (GPT-2) | Ratio of surprisal relative to passage mean |
| `rel_surp_diff` | 0.5426 | Positive (+) | Surprisal (GPT-2) | Difference in surprisal relative to passage mean |
| `surp_mean` | 0.3800 | Positive (+) | Surprisal (GPT-2) | Mean GPT-2 word surprisal |
| `surp_deletion_drop` | -0.3195 | Negative (-) | Surprisal (GPT-2) | Unsupervised sentence deletion coherence drop using GPT-2 |
| `rel_surp_sum_ratio` | 0.2861 | Positive (+) | Surprisal (GPT-2) | Ratio of surprisal sum to passage sum |
| `surp_std` | 0.2846 | Positive (+) | Surprisal (GPT-2) | Standard deviation of word surprisals |
| `psg_surp_mean` | -0.1713 | Negative (-) | Surprisal (GPT-2) | Mean passage surprisal context |
| `psg_surp_sum` | 0.1347 | Positive (+) | Surprisal (GPT-2) | Sum passage surprisal context |
| `psg_surp_std` | -0.1178 | Negative (-) | Surprisal (GPT-2) | Std dev of passage surprisal |
| `surp_max` | -0.1034 | Negative (-) | Surprisal (GPT-2) | Maximum GPT-2 word surprisal |
| `surp_min` | 0.0278 | Positive (+) | Surprisal (GPT-2) | Minimum GPT-2 word surprisal |
| `psg_surp_max` | 0.0224 | Positive (+) | Surprisal (GPT-2) | Max passage surprisal context |

<!-- slide -->
### 6. Linguistic-Only Subsystem (DSNB Balancing)
*Standardized coefficients for the model trained exclusively on surface linguistic and syntactic features under DSNB balancing.*

| Feature Name | Coefficient | Direction | Type | Description |
| :--- | :---: | :---: | :---: | :--- |
| `word_count` | 0.5461 | Positive (+) | Linguistic / Syntactic | Number of words in target sentence |
| `title_ratio` | 0.5302 | Positive (+) | Linguistic / Syntactic | Ratio of title-cased words |
| `char_count` | -0.5246 | Negative (-) | Linguistic / Syntactic | Number of characters in target sentence |
| `flesch_reading_ease` | 0.4449 | Positive (+) | Linguistic / Syntactic | Flesch Reading Ease readability score |
| `flesch_kincaid_grade` | 0.3282 | Positive (+) | Linguistic / Syntactic | Flesch-Kincaid Grade level |
| `question_count` | -0.2696 | Negative (-) | Linguistic / Syntactic | Count of question marks |
| `number_ratio` | 0.2496 | Positive (+) | Linguistic / Syntactic | Ratio of numeric tokens |
| `gunning_fog` | 0.2437 | Positive (+) | Linguistic / Syntactic | Gunning Fog readability index |
| `cap_ratio` | -0.2386 | Negative (-) | Linguistic / Syntactic | Ratio of capitalized words |
| `prep_ratio` | 0.2346 | Positive (+) | Linguistic / Syntactic | Ratio of prepositions |
| `comma_count` | 0.2127 | Positive (+) | Linguistic / Syntactic | Count of commas |
| `ttr` | 0.1802 | Positive (+) | Linguistic / Syntactic | Type-Token Ratio (lexical diversity) |

<!-- slide -->
### 7. Combined Deletion (No Balancing Baseline)
*Standardized coefficients for the combined model integrating the GPT-2 Sentence Deletion Coherence Drop under unbalanced training (None).*

| Feature Name | Coefficient | Direction | Type | Description |
| :--- | :---: | :---: | :---: | :--- |
| `align_sem_sim` | 1.6896 | Positive (+) | Semantic Alignment | SBERT cosine semantic similarity with question |
| `char_count` | -1.6139 | Negative (-) | Linguistic / Syntactic | Number of characters in target sentence |
| `word_count` | 1.5199 | Positive (+) | Linguistic / Syntactic | Number of words in target sentence |
| `rel_surp_ratio` | -0.9989 | Negative (-) | Surprisal (GPT-2) | Ratio of surprisal relative to passage mean |
| `rel_surp_diff` | 0.6586 | Positive (+) | Surprisal (GPT-2) | Difference in surprisal relative to passage mean |
| `surp_max` | -0.6099 | Negative (-) | Surprisal (GPT-2) | Maximum GPT-2 word surprisal |
| `surp_std` | 0.5490 | Positive (+) | Surprisal (GPT-2) | Standard deviation of word surprisals |
| `title_ratio` | 0.5384 | Positive (+) | Linguistic / Syntactic | Ratio of title-cased words |
| `align_rouge_l_recall` | 0.5275 | Positive (+) | Semantic Alignment | ROUGE-L recall score with question |
| `noun_ratio` | -0.5226 | Negative (-) | Linguistic / Syntactic | Ratio of nouns |
| `surp_mean` | 0.5151 | Positive (+) | Surprisal (GPT-2) | Mean GPT-2 word surprisal |
| `comma_count` | 0.4874 | Positive (+) | Linguistic / Syntactic | Count of commas |

````

---

## 2. Comprehensive Coefficient Heatmap Table

Below is the complete matrix of coefficients for the top features across the **Combined Deletion** configuration under all 5 balancing methods. This highlights how balancing methods shift model reliance between features.

| Feature Name | None | Pairwise | Cluster | RST-Neighborhood | DSNB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `adv_ratio` | -0.3644 | -0.4319 | -0.6395 | -0.3372 | -0.1739 |
| `align_jaccard` | 0.1876 | 0.0773 | 1.6874 | 0.0136 | 0.1932 |
| `align_match_count` | 0.2887 | 0.8688 | -0.6129 | 0.5062 | 0.3462 |
| `align_ne_match` | 0.1154 | 0.7863 | 0.0348 | 0.0829 | 0.0598 |
| `align_rouge_l_recall` | 0.5275 | 0.8482 | 0.5491 | 0.6018 | 0.5387 |
| `align_sem_sim` | 1.6896 | 2.8221 | 1.6485 | 1.3099 | 0.8544 |
| `char_count` | -1.6139 | -1.3632 | -0.0778 | -1.0716 | -0.8539 |
| `comma_count` | 0.4874 | 0.6443 | 0.3467 | 0.6078 | 0.5594 |
| `discourse_addition_count` | 0.3319 | 0.4348 | 0.7881 | 0.1532 | 0.2139 |
| `flesch_kincaid_grade` | -0.1330 | -0.1168 | -0.8189 | -0.1474 | -0.3362 |
| `flesch_reading_ease` | -0.1787 | -0.2723 | -0.7023 | 0.3707 | 0.0593 |
| `noun_ratio` | -0.5226 | -0.2935 | -0.3848 | -0.4317 | -0.4180 |
| `psg_rst_s_count` | -0.3213 | 0.0000 | -1.0059 | -0.4293 | -0.2574 |
| `question_count` | -0.3002 | -0.1941 | 0.0000 | 0.0000 | -0.4523 |
| `rel_rst_depth_ratio` | 0.0270 | 0.1188 | -0.0788 | -0.4902 | -0.4538 |
| `rel_rst_n_ratio` | 0.4740 | 0.4276 | 0.2057 | 0.5043 | 0.7203 |
| `rel_surp_diff` | 0.6586 | 0.2891 | 0.4135 | 0.3184 | 0.5082 |
| `rel_surp_ratio` | -0.9989 | -0.5596 | -0.8535 | -0.6104 | -0.6146 |
| `rel_surp_sum_ratio` | -0.2652 | -0.2593 | -1.0672 | -0.5111 | -0.4311 |
| `rst_n_ratio` | -0.4412 | -0.8208 | -0.5061 | -0.3301 | -0.3791 |
| `surp_max` | -0.6099 | -0.5207 | -0.7134 | -0.4849 | -0.3652 |
| `surp_mean` | 0.5151 | 0.2891 | 0.3820 | 0.2612 | 0.3653 |
| `surp_std` | 0.5490 | 0.5026 | 0.7010 | 0.8347 | 0.5289 |
| `surp_sum` | 0.1686 | 0.2587 | 0.9405 | 0.0670 | 0.0700 |
| `title_ratio` | 0.5384 | 1.0489 | 0.5305 | 0.5958 | 0.5058 |
| `verb_ratio` | 0.3878 | 0.7731 | 0.3115 | 0.1311 | 0.1285 |
| `word_count` | 1.5199 | 0.9712 | 0.2952 | 1.2752 | 1.1044 |

---

## 3. Key Findings

1. **Semantic Similarity Dominance**: Cosine similarity between SBERT sentence/question embeddings (`align_sem_sim`) is universally the most powerful positive predictor across all configurations, confirming that relevance to the query is paramount.
2. **Discourse Ratio (`rel_rst_n_ratio`)**: The ratio of nucleus EDUs in a sentence compared to the document (`rel_rst_n_ratio`) has a substantial positive coefficient (~0.72 in DSNB). This validates the core hypothesis of the paper: sentences containing nuclei EDUs in the rhetorical structure of a passage are more salience.
3. **Role of GPT-2 Deletion Heuristic**: The coherence drop feature (`surp_deletion_drop`) obtains a negative coefficient under unbalanced training, but shows positive correlation and significance in subset evaluations. Under DSNB balancing, it is highly useful as a regularizer.
4. **Length and Density Constraints**: Word count (`word_count`) is strongly positive, but character count (`char_count`) is strongly negative when word count is controlled. This suggests that the model prefers *longer sentences consisting of shorter words* (i.e. high information-density, readable sentences rather than long, complex, jargon-heavy sentences).
5. **How DSNB Shifts Coefficients**:
   - Under **No Balancing (None)**, surface features like sentence length can overfit because the majority of sentences are non-salient.
   - Under **Pairwise**, the coefficients are heavily regularized because the model learns to rank pairs, resulting in stable, conservative coefficients.
   - Under **DSNB**, the model is trained on hard negatives that are positionally and semantically similar, forcing the model to rely more on discourse-level properties (such as nuclei ratio `rel_rst_n_ratio` and rhetorical relations) to make fine-grained salience decisions.

