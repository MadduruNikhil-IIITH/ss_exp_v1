# SQuAD Sentence Salience - Discourse & Semantic Inference System

This repository contains the implementation, features, and comparative models for predicting sentence salience in SQuAD context passages. The system integrates Rhetorical Structure Theory (RST) discourse hierarchies, SBERT semantic alignment, GPT-2 lexical surprisals, and syntactic indicators.

---

## 1. Project Overview

Predicting which sentences in a passage are salient (contain answers to specific questions) is a crucial step in reading comprehension, document summarization, and query-focused retrieval. 

This codebase evaluates **13 model configurations** across **5 dataset balancing techniques**, comparing traditional linear models with hybrid deep learning architectures on SQuAD v1.1.

---

## 2. Documentation Index

Detailed guides detailing each step of the data and modeling process are available:
* **[Data Cleaning & Labeling Guide](docs/data_cleaning_process.md)**: Explains the exact-index silver annotation mapping, LLM-as-a-judge verification results, and recommendations for removing boundary noise.
* **[Data Balancing Guide](docs/data_balancing_process.md)**: Outlines the mathematical formulations of the 5 training balancing methods (None, Pairwise, Cluster, RST-Neighborhood, DSNB).
* **[Classifier Catalog](docs/classifier_catalog.md)**: Describes the architecture and parameters of the 13 rule-based, linear, and hybrid transformer models.
* **[Feature Subsets Guide](docs/feature_subsets_guide.md)**: Directory of all 71 extracted features and their configuration mapping.
* **[LLM Verification Report](docs/llm_judge_verification.md)**: Updated results of the local Qwen-1.5B dataset audit.
* **[Feature Importance Report](docs/feature_importance.md)**: Standardized coefficient analysis of all Logistic Regression classifiers.

---

## 3. Core Experimental Results

Evaluated on SQuAD contexts (2,301 training records and 229 validation records) across all 5 dataset balancing techniques. The complete, detailed metrics grid containing Accuracy, Precision, Recall, F1, MRR, MAP, and NDCG for all 13 configurations is available in the dedicated **[Metrics Table (metrics.md)](metrics.md)**.

### Top-Performing Configurations Summary (Threshold = 0.30)

| Model Configuration | Balancing Method | Accuracy | F1-Score | MAP | MRR | NDCG |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gated BERT (Context)** | Pairwise | 0.2882 | 0.3347 | **0.6415** | **0.6408** | **0.7300** |
| **Heuristic-Guided BERT** | DSNB | 0.1921 | 0.3071 | **0.6176** | **0.6165** | **0.7111** |
| **Combined Logistic Regression** | DSNB | 0.2751 | 0.3141 | 0.5617 | 0.5606 | 0.6684 |
| **LGSM** | None | **0.6332** | **0.3438** | 0.5118 | 0.5118 | 0.6320 |

*For the full 65-row results grid, please refer to the **[Complete Experimental Results (metrics.md)](metrics.md)**.*

---

## 4. Dataset Statistics & Analysis

Rigorous statistical analysis of our processed SQuAD sentence-level dataset (2,530 records total) highlights several key characteristics:

### A. Dimensions & Splits
* **Total unique contexts**: 100 (90 train, 10 validation)
* **Total QA pairs**: 491 (452 train, 39 validation)
* **Total sentence-question records**: 2,530 (2,301 train, 229 validation)
* **Class Imbalance**: Salient sentences (Class 1) comprise **19.49%** (493 records), while Non-salient sentences (Class 0) comprise **80.51%** (2,037 records).

### B. Extreme Positional Bias
Over **66.37%** of all salient sentences reside at Sentence Index 0, 1, or 2 inside their context paragraph. This creates a spatial shortcut that classifiers can easily exploit unless neutralized by balancing methods like **DSNB** (Discourse-Semantic Neighborhood Balancing).

![Positional Bias](docs/images/positional_bias.png)

### C. Advanced Statistical Testing (Welch's T-Test)
To establish the validity of our feature set, we conducted Welch's t-tests comparing salient vs. non-salient sentence characteristics:
* **Syntactic Complexity**: Salient sentences have significantly deeper dependency parse trees (mean **6.57** vs. **6.16**, $p < 0.0001$) and larger child-to-head token distances (mean **3.32** vs. **3.06**, $p < 0.0001$), as visualized below:

![Syntactic Complexity](docs/images/syntactic_complexity.png)

* **Information Theoretic (Surprisal)**: Removing salient sentences causes a significantly larger surprisal deletion coherence drop (higher surprisal increase) in the paragraph than removing non-salient sentences ($p < 0.05$), proving they carry critical information context.

![Surprisal Distributions](docs/images/surprisal_distribution.png)

* **Readability Insignificance**: Readability metrics (Flesch Reading Ease and Gunning Fog index) show no statistically significant difference between salient and non-salient sentences, indicating basic text readability is not a strong salience discriminator.

![Readability comparison](docs/images/readability_comparison.png)

### D. Feature Co-linearity Heatmap
Pearson correlation matrix of the top features shows that semantic alignment features (SBERT, Jaccard, ROUGE-L) are highly collinear ($r > 0.85$), whereas RST discourse structures and surprisal features are independent ($r < 0.10$), indicating they provide complementary structural contexts.

![Feature Correlation Heatmap](docs/images/correlation_heatmap.png)

*For more details, see the complete [SQuAD Silver Data Rigorous Statistical Analysis Guide](docs/silver_data_analysis.md) and [Dataset Audit & Analysis Report](docs/dataset_analysis_report.md).*

---

## 5. Running the 4-Stage Pipeline

The codebase is organized into a clean 4-stage execution sequence:

### Step 1: Installation & Setup
Set up a python environment (e.g., Conda) and install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2: Pipeline Execution

#### Stage 1: Dataset Preparation & LLM-as-a-Judge Audit
Prepares SQuAD context splits and runs a local LLM Judge to verify silver target labels:
```bash
python run_stage1_dataset_prep.py --train_ratio 0.9 --total_contexts 100
```

#### Stage 2: Feature Extraction
Extracts concreteness, sentiment, surprisals (UID), syntax, and discourse (RST) features, saving to `features_cache.pkl`:
```bash
python run_stage2_feature_extraction.py
```

#### Stage 3: Model Training & Checkpointing Experiments
Trains all baseline classifiers, Hybrid BERT models, and the LGSM sequence model, saving weights to `checkpoints/`:
```bash
python run_stage3_experiments.py
```

#### Stage 4: Diagnostics, Independent Evaluation & Cross-Validation
*   **Part 1: Standalone Evaluation**: Evaluates saved checkpoints at a specific threshold (e.g., 0.35):
    ```bash
    python run_stage4_evaluation.py --threshold 0.35
    ```
*   **Part 2: Diagnostics Plotting**: Sweeps thresholds and generates Top-K recall curves in `docs/images/`:
    ```bash
    python run_stage4_diagnostics.py
    ```
*   **Part 3: 5-Fold Cross-Validation**: Runs context-level cross-validation to assess generalization variance:
    ```bash
    python run_stage4_cross_validation.py
    ```

---

## 6. Slide Presentations & Academic Reports

*   **LaTeX Beamer Slides**: Source code and compiled PDF slides are located in **`ppt/`** (`ppt/presentation.tex` and `ppt/presentation.pdf`).
*   **SHAP & Positional Recall Analysis**: Explained in [docs/shap_and_lost_in_the_middle.md](docs/shap_and_lost_in_the_middle.md).
*   **BERT Configurations Guide**: Outlined in [docs/hybrid_bert_comparison.md](docs/hybrid_bert_comparison.md).
