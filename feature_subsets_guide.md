# SQuAD Sentence Salience - Feature Subsets Guide

This document lists all 71 features extracted in our pipeline, detailing how heuristic and non-explanatory features are computed, what fallbacks are used when they are not present, and how they map to each model configuration.

---

## 1. Complete Feature Directory

The features are grouped into four primary categories + two heuristic priors:

### A. Linguistic & Readability Features (35 features)
* **Basic Lexical**: `word_count`, `char_count`, `avg_word_length`, `ttr` (Type-Token Ratio), `stopword_ratio`.
* **Casing & Structure**: `cap_ratio`, `title_ratio`, `number_ratio`, `parenthesis_count`.
* **Punctuation**: `comma_count`, `period_count`, `exclamation_count`, `question_count`, `semicolon_count`, `colon_count`.
* **POS Ratios**: `noun_ratio`, `verb_ratio`, `adj_ratio`, `adv_ratio`, `pron_ratio`, `prep_ratio`, `conj_ratio`.
* **Pronoun Person**: `pron_1st_ratio`, `pron_2nd_ratio`, `pron_3rd_ratio` (captures narrative stance).
* **Verb Tense**: `past_tense_ratio`, `present_tense_ratio` (captures temporal context).
* **Syntactic Complexity**:
  * `max_parse_depth`: The maximum depth of the sentence's constituency/dependency parse tree. High depth indicates complex nested clauses.
  * `avg_dep_distance`: The average linear distance (in tokens) between words and their syntactic heads in the dependency tree. Longer distances signify higher processing load.
  * *If not present / calculation fails*: Both default to `0.0`.
* **Lexical Discourse Markers**: `discourse_causal_count`, `discourse_contrast_count`, `discourse_addition_count`.

### B. Surprisal Features (14 features)
Calculated using a causal language model (`gpt2`) in-context. Surprisal of token $w_i$ is computed in bits: $S(w_i) = -\log_2 P(w_i \mid w_{<i})$.
* **Sentence Surprisals**: `surp_mean`, `surp_max`, `surp_min`, `surp_sum`, `surp_std`.
* **Passage Surprisals**: `psg_surp_mean`, `psg_surp_sum`, `psg_surp_max`, `psg_surp_min`, `psg_surp_std`.
  * These represent baseline surprisal values calculated over the entire context paragraph.
* **Relative Surprisals**:
  * `rel_surp_diff`: Sentence mean surprisal minus passage mean surprisal ($\text{mean}_s - \text{mean}_p$).
  * `rel_surp_ratio`: Ratio of sentence mean to passage mean ($\frac{\text{mean}_s}{\text{mean}_p}$).
  * `rel_surp_sum_ratio`: Proportion of the passage's total surprisal contained within the sentence ($\frac{\text{sum}_s}{\text{sum}_p}$).
  * *If not present / calculation fails*: Defaults to `0.0` for sentence metrics, `0.0` for differences, and `1.0` for ratios.

### C. Rhetorical Structure Theory (RST) Features (18 features)
Extracted using `isanlp_rst_v3` to map the passage's discourse tree structure:
* **Subtree Densities**: 
  * `rst_edu_count`: Total number of Elementary Discourse Units (EDUs) mapped to the sentence.
  * `rst_n_count`: Nucleus EDUs count in the sentence. Nucleus units carry the main communicative goal.
  * `rst_s_count`: Satellite EDUs count. Satellite units carry auxiliary info.
  * `rst_n_ratio`: Proportion of nucleus EDUs in the sentence ($\frac{\text{nuclei}_s}{\text{EDUs}_s}$). Defaults to `0.5` if no EDUs exist.
* **Tree Depth & Roots**:
  * `rst_mean_depth`: Average depth of the sentence's EDUs in the document's discourse tree (root is depth 0). Deeper nesting indicates satellites of satellites (nested details). Defaults to `0.0` if not present.
  * `rst_is_root`: Binary flag indicating if any EDU in the sentence is the absolute root of the document's RST tree. Defaults to `0.0`.
* **Passage-level Stats**: `psg_rst_max_depth`, `psg_rst_n_count`, `psg_rst_s_count`.
* **Relative Tree Stats**:
  * `rel_rst_depth_ratio`: Sentence mean depth divided by maximum passage tree depth ($\frac{\text{mean\_depth}_s}{\text{max\_depth}_p}$). Defaults to `0.0`.
  * `rel_rst_n_ratio`: Proportion of document nuclei contained in the sentence ($\frac{\text{nuclei}_s}{\text{nuclei}_p}$). Defaults to `0.0`.
* **RST Relation Frequencies**: Counts of specific relation types (Elaboration, Attribution, Background, Cause, Result, Contrast, Joint) occurring inside or attached to the sentence. Defaults to `0.0`.

### D. Question-Sentence Alignment Features (5 features)
* **Keyword Overlap**: `align_jaccard` (lemma Jaccard coefficient), `align_match_count`, `align_rouge_l_recall`.
* **Expected Named Entity Match (`align_ne_match`)**:
  * Checks if the sentence contains Named Entities that match the expected entity type of the question query-word:
    * Questions containing *Who/Whom/Whose* expect `PERSON` or `ORG` tags.
    * Questions containing *When/Date/Year* expect `DATE` or `TIME` tags.
    * Questions containing *Where/Place/City/Country/State* expect `GPE`, `LOC`, or `FAC` tags.
    * Questions containing *How many/How much/Number/Amount* expect `CARDINAL`, `QUANTITY`, `MONEY`, or `PERCENT` tags.
  * Returns `1.0` if there is a match, else `0.0`. Defaults to `0.0` if parsing fails.
* **Semantic Vector Similarity**: `align_sem_sim` (cosine similarity of Sentence-BERT embeddings). Defaults to `0.0` if not present.

### E. Heuristic Priors (2 features)
* **`rst_rule_based_score`**: 
  * Output probability of the rule-based RST scorer, calculated as a squashed sigmoid weighted sum:
    \[\text{score} = \sigma\left( 2.0 \cdot x_{\text{rst\_is\_root}} + 1.5 \cdot x_{\text{rst\_n\_ratio}} + 1.0 \cdot x_{\text{contrast}} + 1.0 \cdot x_{\text{cause}} - 0.5 \cdot x_{\text{rel\_rst\_depth\_ratio}} \right)\]
    where $\sigma(z) = 1/(1+e^{-z})$. A higher score indicates that the sentence is central and structurally crucial.
  * *If not present / calculation fails*: Defaults to `0.5` (the sigmoid neutral value).
* **`surp_deletion_drop`**: 
  * Unsupervised sentence deletion coherence drop using GPT-2. Measures the average log-probability of all subsequent context tokens (the suffix context) in the original passage versus a modified passage where the target sentence has been deleted:
    \[\text{drop} = \text{Mean}(\log P(\text{suffix} \mid \text{prefix} + \text{sentence})) - \text{Mean}(\log P(\text{suffix} \mid \text{prefix}))\]
    A positive drop indicates that the target sentence provides essential context that makes the rest of the passage significantly more predictable.
  * *If not present / calculation fails*: Defaults to `0.0`.

---

## 2. Feature Mapping by Model Configuration

The table below defines exactly which feature subsets are fed into each of our 13 model configurations.

| Configuration | Text (BERT) | Linguistic (35) | Surprisal (14) | RST (18) | Alignment (5) | Heuristic Priors |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. RST Rule-Based** | ❌ | ❌ | ❌ | ❌ | ❌ | Uses internal rules on RST structure |
| **2. LR (Rst)** | ❌ | ❌ | ❌ | **18 features** | ❌ | ❌ |
| **3. LR (Linguistic)** | ❌ | **35 features** | ❌ | ❌ | ❌ | ❌ |
| **4. LR (Surprisal)** | ❌ | ❌ | **14 features** | ❌ | ❌ | ❌ |
| **5. LR (Combined)** | ❌ | **35 features** | **14 features** | **18 features** | **5 features** | Includes `surp_deletion_drop` |
| **6. Gated BERT (All)** | **BERT text** | **35 features** | **14 features** | **18 features** | **5 features** | Includes `surp_deletion_drop` |
| **7. FiLM BERT (Forced RST)**| **BERT text** | **35 features** | **14 features** | **18 features (FiLM)** | **5 features** | Includes `surp_deletion_drop` |
| **8. Concat BERT (Direct)** | **BERT text** | **35 features** | **14 features** | **18 features** | **5 features** | Includes `surp_deletion_drop` |
| **9. Gated BERT (No RST)** | **BERT text** | **35 features** | **14 features** | ❌ | **5 features** | Includes `surp_deletion_drop` |
| **10. LR (Combined Heu)** | ❌ | **35 features** | **14 features** | **18 features** | **5 features** | `surp_deletion_drop` + `rst_rule_based_score` |
| **11. Guided BERT (RST)** | **BERT text** | **35 features** | **14 features** | **18 features** | **5 features** | `rst_rule_based_score` (fused in head) |
| **12. LR (Combined Del)** | ❌ | **35 features** | **14 features** | **18 features** | **5 features** | `surp_deletion_drop` (fit standardly) |
| **13. Guided BERT (Del)** | **BERT text** | **35 features** | **14 features** | **18 features** | **5 features** | `surp_deletion_drop` (fused in head) |
