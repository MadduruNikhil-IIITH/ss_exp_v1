# Academic Publication Roadmap

This roadmap outlines the strategic steps, paper sections, and scaling actions required to transform this sentence salience experiment into a high-quality academic publication (target venues: ACL, EMNLP, NAACL, or Coling).

---

## 1. Paper Outline & Structure

### A. Title Ideas
- *Idea 1*: "Discourse-Semantic Neighborhood Balancing for Sentence Salience in Extractive Question Answering"
- *Idea 2*: "Rhetorical Structure Priors for Class-Balanced Sentence Salience Ranking"
- *Idea 3*: "Bridging Rhetorical Structure Theory and Transformer Gating for Extractive Question Answering"

### B. Abstract
- **Context**: Extractive QA and Question Generation (QG) pipelines rely on sentence selection to prune non-salient contexts.
- **Problem**: Sentence-level datasets are naturally highly imbalanced (~85% non-salient), causing selectors to overfit or yield high recall with poor precision.
- **Method**: We systematically compare five balancing strategies, including novel discourse-guided undersampling (RST-Neighborhood) and hybrid discourse-semantic neighborhood balancing (DSNB).
- **Results**: Present the key findings comparing tabular baselines, pairwise RankNet architectures, and our heuristic-guided priors. Show that DSNB/Pairwise balancing yields high-precision ranking (MAP/NDCG) optimal for downstream QG.

### C. Introduction
- Frame sentence salience as the bottleneck for long-context LLM pruning and question generation.
- Discuss how standard random undersampling loses contextual cohesion.
- Introduce Rhetorical Structure Theory (RST) as a structural prior to discover "hard negatives" (sentences that are positionally and rhetorically close to the answer span but do not contain it).

### D. Methodology
- **Feature Extraction**: Detail the 71 multi-modal features (35 linguistic, 12 surprisal, 18 RST, 6 SBERT alignment).
- **Class Balancing Flows**:
  - Formalize Cluster-based, RST-Neighborhood, and DSNB formulations.
  - Describe the pairwise delta feature vector formulation ($x_1 - x_2$) and the neural RankNet difference loss.
- **Model Architectures**:
  - Explain Gated BERT, Concat BERT, and FiLM Gated BERT.
  - Detail our new **Heuristic-Guided BERT** that uses the rule-based RST score as a direct structural prior in the classification head.

### E. Experimental Evaluation & Results
- Present the 5-way comparative results table.
- Highlight the trade-offs:
  - **Ranking vs. Classification**: Show that models trained on raw unbalanced data have high accuracy but lower recall, whereas balanced splits (like DSNB or Pairwise) achieve superior ranking capacity (MAP/NDCG) and recall.
  - **Downstream QG Optimization**: High precision (e.g. `70.8%` for Hybrid Gated BERT on Pairwise) is crucial to prevent QG from generating questions from background noise.

---

## 2. Best Course of Action to Maximize Paper Quality

### A. Scaling Up the Dataset
- Our baseline run uses **60 train contexts** and **15 validation contexts** (total 3,478 pairs).
- **For Publication**: Scale the pipeline to **500+ train contexts** and **100+ validation contexts** using [run_feature_extraction.py](file:///d:/Research/Sqaud-Salience/run_feature_extraction.py). This will increase training data to ~25,000+ pairs, providing highly stable, publishable neural results.

### B. Downstream Evaluation (The "Killer" Experiment)
- To prove that sentence salience improves Question Generation:
  1. Train a standard QG model (e.g., T5-base) on SQuAD.
  2. Filter context sentences using our top-performing salience selectors (e.g., Heuristic-Guided BERT).
  3. Generate questions and evaluate them using BLEU, ROUGE, and METEOR.
  4. Show that filtering with our discourse-aware models (DSNB / Heuristic-Guided) produces significantly better questions than random selection or TF-IDF baselines.

### C. Human Evaluation
- Conduct a small human evaluation (2-3 annotators) on 100 generated questions to grade:
  - **Grammaticality**
  - **Answerability** (is the question answerable from the selected sentence?)
  - **Redundancy** (does it repeat context?)
