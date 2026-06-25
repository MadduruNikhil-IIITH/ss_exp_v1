# SQuAD Sentence Salience - Data Cleaning & Labeling Process

This document details the data preparation, silver labeling, and quality verification processes used in our experiments, along with actionable recommendations to enhance label quality.

---

## 1. Silver Label Generation Pipeline

SQuAD v1.1 contains question-answer pairs linked to context passages. Answers are represented as character offsets (`answer_start`) and raw texts (`text`) relative to the context.

To convert this into a sentence-level salience classification dataset, we apply the following process in `src/data_processing.py`:
1. **Sentence Boundary Detection**: We parse the context passage using spaCy's `sentencizer` to extract sentence strings and record their character start/end boundaries (`[start_char, end_char]`) relative to the context.
2. **Exact-Index Span Intersection**: A sentence $s$ is labeled as **salient (Class 1)** if its boundaries overlap with the character span of any annotated answer $[ans\_start, ans\_end]$:
   $$\max(start\_char_s, ans\_start) < \min(end\_char_s, ans\_end)$$
   If no overlap exists, the sentence is labeled as **non-salient (Class 0)**.
3. **Soft Labels Generation**:
   * *Distance-Based Decay*: Assigns scores that decay exponentially with distance from the closest salient sentence: $\text{soft\_label\_decay} = 0.5^{\text{dist}}$.
   * *Hybrid Semantic*: Combines distance decay ($70\%$) and question-sentence TF-IDF similarity ($30\%$).

---

## 2. LLM-as-a-Judge Verification

To evaluate the quality of our exact-index silver labels, we run a local LLM (`Qwen/Qwen2.5-1.5B-Instruct` on GPU) on a balanced sample of 100 sentences. Qwen is prompted to read the context, question, and target sentence, output a reasoning sentence, and render a final judgment (`Yes` or `No`).

* **Verification Metrics**:
  * **Agreement (Accuracy)**: `82.00%`
  * **Cohen's Kappa Score**: `0.6400` (Substantial Agreement)
  * **Precision** (treating LLM as ground truth): `74.00%`
  * **Recall**: `88.10%`
  * **F1-Score**: `80.43%`

---

## 3. Discovered Label Noise and Error Categories

The $18\%$ disagreement between silver labels and the LLM judge exposes two main types of label noise:

### Category A: Silver Salient (1) but LLM Non-Salient (0) [False Positives]
* **Boundary Intersection Noise**: SQuAD answer spans occasionally spill over sentence boundaries by a single character or punctuation mark. The exact-index mapping labels the adjacent sentence as salient, even though it contains no meaningful answer content.
* **Information Insufficiency**: A sentence contains the answer string but lacks the necessary context to answer the question. For example, if the answer is the name "Mathew Knowles" and a sentence mentions *"Mathew Knowles was there"*, it is labeled as Class 1 by offset mapping, but the LLM correctly rejects it because it doesn't answer the specific question *"Mathew Knowles worked for what company?"*.

### Category B: Silver Non-Salient (0) but LLM Salient (1) [False Negatives]
* **Paraphrase Omission**: SQuAD annotators only highlight the specific span containing the short fact. If the passage repeats the answer in a paraphrased sentence, that second sentence is labeled as Class 0 by offset mapping. The LLM correctly identifies it as salient.
* **Coreference Resolution Failure**: Context sentences containing pronouns (e.g. *"He took over the position on July 1, 2005"*) are labeled as Class 0 because the name "John Jenkins" resides in a prior sentence. The LLM resolves this coreference and marks the sentence as salient.

---

## 4. Actionable Recommendations for Silver Data Cleaning

To improve dataset quality for future experiments, we propose the following data cleaning and enhancement steps:

1. **Token-Level Intersection Filter**:
   * *Logic*: Rather than simple character intersection, require the overlapping span between the sentence and the answer to contain at least one non-stopword, alphanumeric token.
   * *Impact*: Eliminates false positives caused by boundary-spilling whitespace or punctuation.
2. **Coreference Resolution Pre-processing**:
   * *Logic*: Integrate a coreference resolution pipeline (e.g., using `fastcoref` or spaCy's coref resolver) before boundary checks. Replace pronouns with their resolved entities to map answer connections.
   * *Impact*: Eliminates false negatives caused by pronominal references.
3. **Cross-Encoder Semantic Alignment**:
   * *Logic*: Use a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to score the entailment/salience of each sentence against the question and the answer text. Relabel sentences with entailment scores $>0.7$ as Class 1.
   * *Impact*: Automatically captures paraphrases and semantic equivalents missed by exact span checks.
