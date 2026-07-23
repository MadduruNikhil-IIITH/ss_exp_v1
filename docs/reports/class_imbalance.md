# SQuAD Sentence Salience Class Imbalance & Data Cleaning Statistics

This report documents the class distribution of the sentence salience dataset before and after applying the five balancing techniques.

---

## 1. SQuAD Paragraph Split vs. Sentence-Level Sizing

To prevent data leakage, the dataset split is strictly performed at the **SQuAD Context Paragraph level**. SQuAD context passages never overlap between the training and validation sets.

However, the classification and ranking unit is the **individual sentence**. For each question associated with a paragraph, we create a record for every sentence in that paragraph. Thus, the total number of evaluation records is calculated as:
$$\text{Total Records} = \sum_{\text{paragraphs}} (\text{Number of Questions} \times \text{Number of Sentences in Paragraph})$$

### Class Distribution Summary Table

| Dataset Split | SQuAD Paragraphs (Contexts) | Balancing State | Salient (Positive Class = 1) / Pairs | Non-Salient (Negative Class = 0) | Positive Class Ratio (%) | Negative Class Ratio (%) | Total Question-Sentence Pairs (or Pairs Generated) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Set** | 60 | **None** (Raw Unbalanced) | 339 | 1,817 | 15.72% | 84.28% | 2,156 |
| **Training Set** | 60 | **Pairwise** | 1,831 pairs | - | 50.00% (Balanced) | 50.00% (Balanced) | 1,831 pairs (3,662 inputs) |
| **Training Set** | 60 | **Cluster** | 339 | 339 | 50.00% | 50.00% | 678 |
| **Training Set** | 60 | **RST-Neighborhood** | 339 | 339 | 50.00% | 50.00% | 678 |
| **Training Set** | 60 | **DSNB** | 339 | 339 | 50.00% | 50.00% | 678 |
| **Validation Set**| 15 | **None** (Natural Dev) | 336 | 986 | 25.42% | 74.58% | 1,322 |

---

## 2. Key Statistical Insights

### A. The SQuAD Class Imbalance Challenge
Sentence salience detection on SQuAD is naturally highly imbalanced. In any reading comprehension passage context, only **one or two sentences** contain the actual answer, while the remaining **5 to 10 sentences** provide background context but are non-salient.
- Before balancing, **84.3%** of the training sentences are non-salient.
- Standard training on this raw distribution causes classifiers (especially neural networks) to default to predicting the majority class (non-salient) to minimize cross-entropy loss, resulting in high accuracy but extremely poor recall and MRR/MAP scores.

### B. Why is the Validation Set Positive Ratio Higher?
A noticeable distribution shift occurs: the raw training split has **15.7%** positive instances, whereas the validation split has **25.4%** positive instances. This is a direct consequence of SQuAD's annotation design:
1. **SQuAD Train set**: Contains exactly **one human-annotated answer** per question.
2. **SQuAD Validation/Dev set**: Contains **three separate human-annotated answers** per question to account for human variation.
3. **Exact-Index Mapping**: Our silver labeling logic marks a sentence as salient if it intersects with **any** annotated answer span. Because the validation set has 3 times more annotated spans, the probability of mapping answers to multiple sentences (due to slightly different character span selections or alternative answers) increases, elevating the proportion of salient sentences.

### C. Class Balancing Methodologies

1. **None**:
   - Retains the raw unbalanced dataset (2,156 training pairs).
   - Serves as the natural baseline.

2. **Pairwise Balancing (Training Only)**:
   - Pairs every positive sentence with a negative sentence from the same context.
   - For a context with $P$ positive sentences and $N$ negative sentences, it generates $P \times N$ pairs.
   - Yields balanced pairs where the model learns relative ranking ($s_1 > s_2$). Under this setting, we train on the difference vector $x_{\text{diff}} = x_1 - x_2$.
   - Generates **1,831 pairs** for our 60-context SQuAD training set.

3. **Cluster-Based Undersampling (Training Only)**:
   - Identifies the $P$ positive samples ($P = 339$).
   - Runs K-Means clustering on the $N$ negative samples ($N = 1,817$) where $K = P = 339$.
   - Selects the single negative sample closest to each of the 339 cluster centers.
   - Saves **678 balanced sentences**, retaining representative context variety while removing redundancy.

4. **RST-Neighborhood Balancing (Training Only)**:
   - A discourse-aware method that selects negative sentences that are positionally or rhetorically closest to the salient sentence using linear context indices and RST tree depths.
   - Selects $P = 339$ hard negatives (total **678 balanced sentences**).

5. **Discourse-Semantic Neighborhood Balancing (DSNB) (Training Only)**:
   - Combines linear position, discourse tree depth similarity, and semantic alignment (SBERT cosine similarity to the question) to choose the hardest negatives.
   - Selects $P = 339$ hard negatives (total **678 balanced sentences**).
