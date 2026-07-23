# SQuAD Sentence Salience - Feature Subsets Guide

This document lists all **78 features** extracted in our streamlined, query-independent pipeline, detailing how structural, lexical, discourse, sentiment, concreteness, and surprisal features are computed, what fallbacks are used, and how they map to each model configuration.

---

## 1. Complete Feature Directory

The features are grouped into three primary categories + one heuristic prior (alignment features have been deprecated and removed to support query-independent operation):

### A. Linguistic, Readability, Sentiment, & Concreteness Features (43 features)
* **Basic Lexical**: `word_count`, `char_count`, `avg_word_length`, `ttr` (Type-Token Ratio), `stopword_ratio`.
* **Casing & Structure**: `cap_ratio`, `title_ratio`, `number_ratio`, `parenthesis_count`.
* **Punctuation**: `comma_count`, `period_count`, `exclamation_count`, `question_count`, `semicolon_count`, `colon_count`.
* **POS Ratios**: `noun_ratio`, `verb_ratio`, `adj_ratio`, `adv_ratio`, `pron_ratio`, `prep_ratio`, `conj_ratio`.
* **Pronoun Person**: `pron_1st_ratio`, `pron_2nd_ratio`, `pron_3rd_ratio` (captures narrative stance).
* **Verb Tense**: `past_tense_ratio`, `present_tense_ratio` (captures temporal context).
* **Syntactic Complexity**:
  * `max_parse_depth`: The maximum depth of the sentence's constituency/dependency parse tree. High depth indicates complex nested clauses.
  * `avg_dep_distance`: The average linear distance (in tokens) between words and their syntactic heads in the dependency tree. Longer distances signify higher processing load.
  * *Fallback*: Both default to `0.0` if constituency parsing fails.
* **Lexical Discourse Markers**: `discourse_causal_count`, `discourse_contrast_count`, `discourse_addition_count`.
* **Brysbaert Concreteness Ratings (4 features)**:
  * `concrete_mean`, `concrete_max`, `concrete_min`, `concrete_std`: Statistical aggregates of concreteness ratings for non-stopwords using Brysbaert concreteness norms (scale of 1-5).
  * *Fallback*: Defaults to `3.0` (neutral) if no rated words are found.
* **VADER Sentiment Polarity (4 features)**:
  * `sentiment_polarity_pos`, `sentiment_polarity_neg`, `sentiment_polarity_neu`, `sentiment_polarity_compound`: Sentiment intensities computed using the VADER Sentiment Intensity Analyzer.

### B. Surprisal Features (17 features)
Calculated using the causal language model (`gpt2`) and masked language model (`bert-base-uncased`) via the **PsychFormers** segmentation setup. Surprisals are calculated in bits: $S(w_i) = -\log_2 P(w_i \mid \text{context})$.
* **PsychFormers Causal Surprisals (8 features)**: `surp_causal_pf_mean`, `surp_causal_pf_max`, `surp_causal_pf_min`, `surp_causal_pf_sum`, `surp_causal_pf_std`, and passage-relative counterparts `rel_surp_causal_pf_diff`, `rel_surp_causal_pf_ratio`, and `rel_surp_causal_pf_sum_ratio`.
* **PsychFormers Masked PLL Coherence (8 features)**: `surp_pll_pf_mean`, `surp_pll_pf_max`, `surp_pll_pf_min`, `surp_pll_pf_sum`, `surp_pll_pf_std`, and passage-relative counterparts `rel_surp_pll_pf_diff`, `rel_surp_pll_pf_ratio`, and `rel_surp_pll_pf_sum_ratio`.
* **Sentence Deletion Coherence Drop (1 feature)**:
  * `surp_deletion_drop`: Unsupervised sentence deletion drop calculated using the PsychFormers causal model. Measures the change in mean token surprisal over the passage when the target sentence is deleted:
    $$\text{drop} = \text{Mean Surprisal}_{\text{GPT-2}}(P \setminus \{s\}) - \text{Mean Surprisal}_{\text{GPT-2}}(P)$$
  * *Fallback*: Defaults to `0.0`.

### C. Rhetorical Structure Theory (RST) Features (18 features)
Mapped from the document's rhetorical discourse tree:
* **Subtree Densities**: `rst_edu_count`, `rst_n_count` (nucleus EDUs), `rst_s_count` (satellite EDUs), and `rst_n_ratio` (proportion of nuclei in the sentence).
* **Tree Depth & Roots**:
  * `rst_mean_depth`: Average depth of the sentence's EDUs in the document's discourse tree (root is depth 0). Deeper nesting indicates satellites of satellites. Defaults to `0.0`.
  * `rst_is_root`: Binary flag indicating if any EDU in the sentence is the absolute root of the document's RST tree. Defaults to `0.0`.
* **Passage-level Stats**: `psg_rst_max_depth`, `psg_rst_n_count`, `psg_rst_s_count`.
* **Relative Tree Stats**: `rel_rst_depth_ratio` and `rel_rst_n_ratio`.
* **RST Relation Frequencies**: Counts of specific relations (Elaboration, Attribution, Background, Cause, Result, Contrast, Joint) occurring inside or attached to the sentence. Defaults to `0.0`.

### D. Question-Sentence Alignment Features (Deprecated & Removed)
*   All question-alignment features (`align_sem_sim`, `align_jaccard`, `align_match_count`, `align_rouge_l_recall`, `align_ne_match`) have been **completely removed** to support pure query-independent salience classification.

### E. Heuristic Prior (1 feature)
* **`rst_rule_based_score`**: Output probability of the rule-based RST scorer, calculated as a squashed sigmoid weighted sum:
  \[\text{score} = \sigma\left( 2.0 \cdot x_{\text{rst\_is\_root}} + 1.5 \cdot x_{\text{rst\_n\_ratio}} + 1.0 \cdot x_{\text{contrast}} + 1.0 \cdot x_{\text{cause}} - 0.5 \cdot x_{\text{rel\_rst\_depth\_ratio}} \right)\]
  Used as a prior probability weight injected directly into the neural logits of **Config 11 (Heuristic-Guided BERT)** during inference.

---

## 2. Feature Mapping by Model Configuration

The table below defines exactly which feature subsets are fed into each of our **11 core configurations**:

| Configuration | Text (BERT) | Linguistic/Conc/Sent (43) | Surprisal (17) | RST (18) | Heuristic Prior / Sequence |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. RST Rule-Based** | ❌ | ❌ | ❌ | ❌ | Uses internal rules on RST structure |
| **2. LR (Rst)** | ❌ | ❌ | ❌ | **18 features** | ❌ |
| **3. LR (Linguistic)** | ❌ | **43 features** | ❌ | ❌ | ❌ |
| **4. LR (Surprisal)** | ❌ | ❌ | **17 features** | ❌ | ❌ |
| **5. LR (Combined)** | ❌ | **43 features** | **17 features** | **18 features** | ❌ |
| **6. Gated BERT (Context)** | **BERT text** | **43 features** | **17 features** | **18 features** | ❌ |
| **7. FiLM BERT (RST)**| **BERT text** | **43 features** | **17 features** | **18 features (FiLM)** | ❌ |
| **8. Concat BERT (Context)** | **BERT text** | **43 features** | **17 features** | **18 features** | ❌ |
| **9. Gated BERT (No RST)** | **BERT text** | **43 features** | **17 features** | ❌ | ❌ |
| **11. Guided BERT (RST)** | **BERT text** | **43 features** | **17 features** | **18 features** | `rst_rule_based_score` (fused in head) |
| **12. LGSM** | **BERT text** | **43 features** | **17 features** | **18 features** | Contextual Transformer Sequence modeling |
