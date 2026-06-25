# SQuAD Sentence Salience - Discourse & Semantic Inference System

This repository contains the implementation, features, and comparative models for predicting sentence salience in SQuAD context passages. The system integrates Rhetorical Structure Theory (RST) discourse hierarchies, SBERT semantic alignment, GPT-2 lexical surprisals, and syntactic indicators.

---

## 1. Project Overview

Predicting which sentences in a passage are salient (contain answers to specific questions) is a crucial step in reading comprehension, document summarization, and query-focused retrieval. 

This codebase evaluates **13 model configurations** across **5 dataset balancing techniques**, comparing traditional linear models with hybrid deep learning architectures on SQuAD v1.1.

---

## 2. Documentation Index

Detailed guides detailing each step of the data and modeling process are available:
* **[Data Cleaning & Labeling Guide](data_cleaning_process.md)**: Explains the exact-index silver annotation mapping, LLM-as-a-judge verification results, and recommendations for removing boundary noise.
* **[Data Balancing Guide](data_balancing_process.md)**: Outlines the mathematical formulations of the 5 training balancing methods (None, Pairwise, Cluster, RST-Neighborhood, DSNB).
* **[Classifier Catalog](classifier_catalog.md)**: Describes the architecture and parameters of the 13 rule-based, linear, and hybrid transformer models.
* **[Feature Subsets Guide](feature_subsets_guide.md)**: Directory of all 71 extracted features and their configuration mapping.
* **[LLM Verification Report](llm_judge_verification.md)**: Updated results of the local Qwen-1.5B dataset audit.
* **[Feature Importance Report](feature_importance.md)**: Standardized coefficient analysis of all Logistic Regression classifiers.

---

## 3. Core Experimental Results

Evaluated on SQuAD contexts (2,156 training records, 1,322 validation records), sorted by model configuration and balancing progression:

| Model Configuration                   | Balancing        |   Accuracy |   Precision |   Recall |       F1 |      MRR |      MAP |     NDCG |
|:--------------------------------------|:-----------------|-----------:|------------:|---------:|---------:|---------:|---------:|---------:|
| 1. RST Rule-Based                     | None             |   0.265507 |    0.249803 | 0.943452 | 0.395016 | 0.538484 | 0.532043 | 0.816837 |
| 2. LR (Rst)                           | None             |   0.490166 |    0.314286 | 0.851190 | 0.459069 | 0.636253 | 0.632163 | 0.851751 |
| 2. LR (Rst)                           | Pairwise         |   0.667171 |    0.348837 | 0.357143 | 0.352941 | 0.637604 | 0.633514 | 0.852411 |
| 2. LR (Rst)                           | Cluster          |   0.464448 |    0.290068 | 0.764881 | 0.420622 | 0.630453 | 0.628422 | 0.850981 |
| 2. LR (Rst)                           | RST-Neighborhood |   0.457640 |    0.281787 | 0.732143 | 0.406948 | 0.598362 | 0.592433 | 0.834977 |
| 2. LR (Rst)                           | DSNB             |   0.431165 |    0.251790 | 0.627976 | 0.359455 | 0.548275 | 0.543701 | 0.807394 |
| 3. LR (Linguistic)                    | None             |   0.426626 |    0.265556 | 0.711310 | 0.386731 | 0.584284 | 0.575086 | 0.829298 |
| 3. LR (Linguistic)                    | Pairwise         |   0.254160 |    0.254160 | 1.000000 | 0.405308 | 0.590264 | 0.581513 | 0.833157 |
| 3. LR (Linguistic)                    | Cluster          |   0.395613 |    0.278469 | 0.866071 | 0.421434 | 0.585270 | 0.576779 | 0.822707 |
| 3. LR (Linguistic)                    | RST-Neighborhood |   0.474281 |    0.292964 | 0.755952 | 0.422278 | 0.571704 | 0.563516 | 0.821941 |
| 3. LR (Linguistic)                    | DSNB             |   0.448563 |    0.271246 | 0.693452 | 0.389958 | 0.591781 | 0.579537 | 0.831176 |
| 4. LR (Surprisal)                     | None             |   0.516641 |    0.320710 | 0.806548 | 0.458933 | 0.619362 | 0.613040 | 0.849838 |
| 4. LR (Surprisal)                     | Pairwise         |   0.745840 |    0.000000 | 0.000000 | 0.000000 | 0.627393 | 0.622309 | 0.853072 |
| 4. LR (Surprisal)                     | Cluster          |   0.530257 |    0.288262 | 0.577381 | 0.384539 | 0.644421 | 0.635871 | 0.858923 |
| 4. LR (Surprisal)                     | RST-Neighborhood |   0.480333 |    0.271782 | 0.622024 | 0.378281 | 0.556357 | 0.545717 | 0.802890 |
| 4. LR (Surprisal)                     | DSNB             |   0.475038 |    0.273990 | 0.645833 | 0.384752 | 0.531581 | 0.525998 | 0.791241 |
| 5. LR (Combined)                      | None             |   0.786687 |    0.561086 | 0.738095 | 0.637532 | 0.888834 | 0.881206 | 0.949559 |
| 5. LR (Combined)                      | Pairwise         |   0.358548 |    0.283051 | 0.994048 | 0.440633 | 0.892959 | 0.886157 | 0.949583 |
| 5. LR (Combined)                      | Cluster          |   0.764750 |    0.535211 | 0.565476 | 0.549928 | 0.858636 | 0.847611 | 0.934179 |
| 5. LR (Combined)                      | RST-Neighborhood |   0.696672 |    0.446982 | 0.815476 | 0.577450 | 0.836634 | 0.830088 | 0.927082 |
| 5. LR (Combined)                      | DSNB             |   0.722390 |    0.471767 | 0.770833 | 0.585311 | 0.814246 | 0.806705 | 0.919504 |
| 6. Hybrid Gated BERT (All Features)   | None             |   0.760968 |    0.523697 | 0.657738 | 0.583113 | 0.851705 | 0.843253 | 0.934516 |
| 6. Hybrid Gated BERT (All Features)   | Pairwise         |   0.252648 |    0.252280 | 0.988095 | 0.401937 | 0.793792 | 0.784359 | 0.917796 |
| 6. Hybrid Gated BERT (All Features)   | Cluster          |   0.624811 |    0.392761 | 0.872024 | 0.541590 | 0.769252 | 0.761294 | 0.901901 |
| 6. Hybrid Gated BERT (All Features)   | RST-Neighborhood |   0.405446 |    0.291667 | 0.937500 | 0.444915 | 0.705614 | 0.694963 | 0.878181 |
| 6. Hybrid Gated BERT (All Features)   | DSNB             |   0.406203 |    0.288407 | 0.910714 | 0.438082 | 0.758172 | 0.745213 | 0.897044 |
| 7. FiLM BERT (Forced RST + Skip Link) | None             |   0.775340 |    0.538614 | 0.809524 | 0.646849 | 0.874147 | 0.866038 | 0.944253 |
| 7. FiLM BERT (Forced RST + Skip Link) | Pairwise         |   0.745840 |    0.000000 | 0.000000 | 0.000000 | 0.842629 | 0.832334 | 0.930793 |
| 7. FiLM BERT (Forced RST + Skip Link) | Cluster          |   0.592284 |    0.343606 | 0.663690 | 0.452792 | 0.642197 | 0.635958 | 0.850580 |
| 7. FiLM BERT (Forced RST + Skip Link) | RST-Neighborhood |   0.542360 |    0.322325 | 0.726190 | 0.446478 | 0.643816 | 0.637485 | 0.854113 |
| 7. FiLM BERT (Forced RST + Skip Link) | DSNB             |   0.404690 |    0.201325 | 0.452381 | 0.278643 | 0.470710 | 0.467806 | 0.783717 |
| 8. Concat BERT (Direct Concatenation) | None             |   0.791225 |    0.579787 | 0.648810 | 0.612360 | 0.844774 | 0.835524 | 0.930115 |
| 8. Concat BERT (Direct Concatenation) | Pairwise         |   0.791225 |    0.576142 | 0.675595 | 0.621918 | 0.849395 | 0.839659 | 0.934936 |
| 8. Concat BERT (Direct Concatenation) | Cluster          |   0.494705 |    0.311791 | 0.818452 | 0.451560 | 0.741124 | 0.731296 | 0.886512 |
| 8. Concat BERT (Direct Concatenation) | RST-Neighborhood |   0.622542 |    0.359240 | 0.619048 | 0.454645 | 0.641910 | 0.637866 | 0.854040 |
| 8. Concat BERT (Direct Concatenation) | DSNB             |   0.666415 |    0.272727 | 0.187500 | 0.222222 | 0.598590 | 0.586424 | 0.829952 |
| 9. Gated BERT (No RST features)       | None             |   0.737519 |    0.490196 | 0.818452 | 0.613155 | 0.816337 | 0.808223 | 0.920809 |
| 9. Gated BERT (No RST features)       | Pairwise         |   0.459153 |    0.311817 | 0.934524 | 0.467610 | 0.839494 | 0.830990 | 0.932316 |
| 9. Gated BERT (No RST features)       | Cluster          |   0.691377 |    0.437063 | 0.744048 | 0.550661 | 0.784158 | 0.776324 | 0.907858 |
| 9. Gated BERT (No RST features)       | RST-Neighborhood |   0.711800 |    0.458716 | 0.744048 | 0.567537 | 0.785259 | 0.776539 | 0.913011 |
| 9. Gated BERT (No RST features)       | DSNB             |   0.634644 |    0.336303 | 0.449405 | 0.384713 | 0.755375 | 0.748837 | 0.897828 |
| 10. LR (Combined Heuristic)           | None             |   0.785930 |    0.559819 | 0.738095 | 0.636714 | 0.888834 | 0.881206 | 0.949557 |
| 10. LR (Combined Heuristic)           | Pairwise         |   0.365356 |    0.284859 | 0.991071 | 0.442525 | 0.892959 | 0.886157 | 0.949579 |
| 10. LR (Combined Heuristic)           | Cluster          |   0.764750 |    0.535211 | 0.565476 | 0.549928 | 0.860836 | 0.850453 | 0.934871 |
| 10. LR (Combined Heuristic)           | RST-Neighborhood |   0.699697 |    0.450082 | 0.818452 | 0.580781 | 0.836744 | 0.830748 | 0.927292 |
| 10. LR (Combined Heuristic)           | DSNB             |   0.724660 |    0.474359 | 0.770833 | 0.587302 | 0.816447 | 0.808906 | 0.920161 |
| 11. Heuristic-Guided BERT (RST)       | None             |   0.828290 |    0.648501 | 0.708333 | 0.677098 | 0.908526 | 0.894738 | 0.953453 |
| 11. Heuristic-Guided BERT (RST)       | Pairwise         |   0.815431 |    0.622995 | 0.693452 | 0.656338 | 0.897745 | 0.885057 | 0.947494 |
| 11. Heuristic-Guided BERT (RST)       | Cluster          |   0.757943 |    0.514760 | 0.830357 | 0.635535 | 0.875523 | 0.863233 | 0.944495 |
| 11. Heuristic-Guided BERT (RST)       | RST-Neighborhood |   0.719365 |    0.469880 | 0.812500 | 0.595420 | 0.874147 | 0.863449 | 0.944667 |
| 11. Heuristic-Guided BERT (RST)       | DSNB             |   0.695159 |    0.326425 | 0.187500 | 0.238185 | 0.599100 | 0.593050 | 0.829715 |
| 12. LR (Combined Deletion)            | None             |   0.786687 |    0.561086 | 0.738095 | 0.637532 | 0.888834 | 0.881206 | 0.949559 |
| 12. LR (Combined Deletion)            | Pairwise         |   0.358548 |    0.283051 | 0.994048 | 0.440633 | 0.894609 | 0.887807 | 0.949953 |
| 12. LR (Combined Deletion)            | Cluster          |   0.765507 |    0.536723 | 0.565476 | 0.550725 | 0.858636 | 0.847611 | 0.934170 |
| 12. LR (Combined Deletion)            | RST-Neighborhood |   0.696672 |    0.446982 | 0.815476 | 0.577450 | 0.836634 | 0.830088 | 0.927020 |
| 12. LR (Combined Deletion)            | DSNB             |   0.724660 |    0.474453 | 0.773810 | 0.588235 | 0.818097 | 0.810006 | 0.920711 |
| 13. Heuristic-Guided BERT (Deletion)  | None             |   0.799546 |    0.707602 | 0.360119 | 0.477318 | 0.878273 | 0.869696 | 0.945702 |
| 13. Heuristic-Guided BERT (Deletion)  | Pairwise         |   0.836611 |    0.701342 | 0.622024 | 0.659306 | 0.892464 | 0.879478 | 0.944366 |
| 13. Heuristic-Guided BERT (Deletion)  | Cluster          |   0.577912 |    0.339595 | 0.699405 | 0.457198 | 0.775554 | 0.764278 | 0.909953 |
| 13. Heuristic-Guided BERT (Deletion)  | RST-Neighborhood |   0.645234 |    0.411215 | 0.916667 | 0.567742 | 0.877833 | 0.868160 | 0.944678 |
| 13. Heuristic-Guided BERT (Deletion)  | DSNB             |   0.679274 |    0.426910 | 0.764881 | 0.547974 | 0.857041 | 0.850000 | 0.939658 |

---

## 4. Quick Start

### Installation
Set up a python environment (e.g. Conda) and install requirements:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Running the Pipeline
Run the full SQuAD feature extraction, model fitting, and evaluation pipeline:
```bash
python run_pipeline.py
```
This script automatically runs all 13 model configurations across all 5 dataset balancing techniques and saves the output to `metrics.csv` and `metrics.md`.
