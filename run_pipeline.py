import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from src.data_processing import apply_pairwise_balancing, apply_cluster_balancing, apply_rst_balancing, apply_dsnb_balancing
from src.classifiers.rule_based_rst import RuleBasedRSTClassifier
from src.classifiers.logistic_reg import TabularClassifierWrapper
from src.classifiers.hybrid_bert import HybridBERTClassifier
from run_stage1 import run_stage1_pipeline

def compute_classification_metrics(y_true, y_pred):
    """
    Computes standard classification metrics.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

def compute_ranking_metrics(records, scores):
    """
    Computes Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), and
    Normalized Discounted Cumulative Gain (NDCG) per question.
    """
    # Group prediction scores by question_id
    q_groups = {}
    for idx, r in enumerate(records):
        q_id = r["question_id"]
        if q_id not in q_groups:
            q_groups[q_id] = []
        q_groups[q_id].append((r["binary_label"], r["soft_label_hybrid"], scores[idx]))
        
    mrr_list = []
    map_list = []
    ndcg_list = []
    
    for q_id, group in q_groups.items():
        # Sort sentences by predicted score descending
        group_sorted = sorted(group, key=lambda x: x[2], reverse=True)
        
        y_true_bin = [x[0] for x in group_sorted]
        y_true_soft = [x[1] for x in group_sorted]
        
        # 1. Reciprocal Rank (RR)
        rr = 0.0
        for rank_idx, val in enumerate(y_true_bin):
            if val == 1:
                rr = 1.0 / (rank_idx + 1)
                break
        mrr_list.append(rr)
        
        # 2. Average Precision (AP)
        num_hits = 0
        sum_precisions = 0.0
        num_relevant = sum(y_true_bin)
        if num_relevant > 0:
            for rank_idx, val in enumerate(y_true_bin):
                if val == 1:
                    num_hits += 1
                    precision_at_k = num_hits / (rank_idx + 1)
                    sum_precisions += precision_at_k
            ap = sum_precisions / num_relevant
            map_list.append(ap)
            
        # 3. NDCG (using continuous soft hybrid labels)
        def dcg(rel):
            return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
            
        rel_scores = y_true_soft
        actual_dcg = dcg(rel_scores)
        ideal_dcg = dcg(sorted(rel_scores, reverse=True))
        ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcg_list.append(ndcg)
        
    return {
        "MRR": np.mean(mrr_list) if mrr_list else 0.0,
        "MAP": np.mean(map_list) if map_list else 0.0,
        "NDCG": np.mean(ndcg_list) if ndcg_list else 0.0
    }

def main():
    cache_path = "features_cache_deletion.pkl"
    
    # Run Stage 1 to build cache if it doesn't exist
    if not os.path.exists(cache_path):
        print("Feature cache not found. Running Stage 1 Feature Extraction Pipeline...")
        # Using 60 training contexts and 15 validation contexts for the scaled experiment
        run_stage1_pipeline(num_train_contexts=60, num_val_contexts=15, cache_path=cache_path)
        
    print(f"Loading cached features from '{cache_path}'...")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
        
    train_records = cache_data["train"]
    val_records = cache_data["validation"]
    
    print(f"Loaded {len(train_records)} training records and {len(val_records)} validation records.")
    
    # Default list of all feature columns for clustering
    if train_records:
        all_features = list(train_records[0]["features"].keys())
    else:
        print("Error: Empty cached records.")
        return

    # Prepare standard (unbalanced) validation ground truth
    y_val_true = np.array([r["binary_label"] for r in val_records])
    
    # Dictionary to collect results
    results = []

    # =========================================================================
    # CONFIG 1: RST Rule-Based Heuristics
    # =========================================================================
    print("\n--- Training Config 1: RST Rule-Based Heuristics ---")
    rb_classifier = RuleBasedRSTClassifier()
    # Extracts rst features dict list
    val_rst_feats = [r["features"] for r in val_records]
    probas = rb_classifier.predict_proba(val_rst_feats)
    preds = rb_classifier.predict(val_rst_feats)
    
    cls_metrics = compute_classification_metrics(y_val_true, preds)
    rank_metrics = compute_ranking_metrics(val_records, probas)
    
    results.append({
        "Model Configuration": "1. RST Rule-Based",
        "Balancing": "None",
        **cls_metrics,
        **rank_metrics
    })

    # Pre-generate balanced training datasets to share between LR and BERT models
    balanced_datasets = {
        "None": train_records,
        "Pairwise": apply_pairwise_balancing(train_records),
        "Cluster": apply_cluster_balancing(train_records, all_features),
        "RST-Neighborhood": apply_rst_balancing(train_records),
        "DSNB": apply_dsnb_balancing(train_records)
    }

    # List of training balancing techniques to compare
    balancing_methods = ["None", "Pairwise", "Cluster", "RST-Neighborhood", "DSNB"]
    
    for balancing in tqdm(balancing_methods, desc="Logistic Regression Experiments"):
        balanced_train = balanced_datasets[balancing]

        # =========================================================================
        # CONFIG 2-5: Logistic Regression Configurations
        # =========================================================================
        modes = ["rst", "linguistic", "surprisal", "combined", "combined_heuristic", "combined_deletion"]
        for mode in modes:
            config_num = {"rst": 2, "linguistic": 3, "surprisal": 4, "combined": 5, "combined_heuristic": 10, "combined_deletion": 12}[mode]
            model_name = f"{config_num}. LR ({mode.replace('_', ' ').title()})"
            print(f"\n--- Training {model_name} with {balancing} balancing ---")
            
            lr_wrapper = TabularClassifierWrapper(feature_mode=mode, use_soft_targets=False)
            lr_wrapper.fit(balanced_train)
            
            probas = lr_wrapper.predict_proba(val_records)
            preds = lr_wrapper.predict(val_records)
            
            cls_metrics = compute_classification_metrics(y_val_true, preds)
            rank_metrics = compute_ranking_metrics(val_records, probas)
            
            results.append({
                "Model Configuration": model_name,
                "Balancing": balancing,
                **cls_metrics,
                **rank_metrics
            })

    # =========================================================================
    # CONFIG 6-9: BERT-Based Gating Configurations
    # =========================================================================
    # We will evaluate BERT models under all 5 balancing configurations:
    # Train the 4 BERT variants on each of the 5 balanced splits.
    bert_experiments = []
    for balancing in ["None", "Pairwise", "Cluster", "RST-Neighborhood", "DSNB"]:
        for mode, name in [
            ("gated_all", "6. Hybrid Gated BERT (All Features)"),
            ("film_rst_skip", "7. FiLM BERT (Forced RST + Skip Link)"),
            ("concat_all", "8. Concat BERT (Direct Concatenation)"),
            ("no_rst", "9. Gated BERT (No RST features)"),
            ("heuristic_guided_rst", "11. Heuristic-Guided BERT (RST)"),
            ("heuristic_guided_deletion", "13. Heuristic-Guided BERT (Deletion)")
        ]:
            bert_experiments.append((mode, name, balancing))
    
    for mode, name, balancing in tqdm(bert_experiments, desc="BERT Experiments"):
        print(f"\n--- Training {name} with {balancing} balancing ---")
        train_data = balanced_datasets[balancing]
        
        bert_classifier = HybridBERTClassifier(mode=mode, device="cuda")
        # Train for 2 epochs for quick verification
        bert_classifier.fit(train_data, val_records, epochs=2, batch_size=32, use_soft_labels=False)
        
        probas = bert_classifier.predict_proba(val_records, batch_size=32)
        preds = bert_classifier.predict(val_records, batch_size=32)
        
        cls_metrics = compute_classification_metrics(y_val_true, preds)
        rank_metrics = compute_ranking_metrics(val_records, probas)
        
        results.append({
            "Model Configuration": name,
            "Balancing": balancing,
            **cls_metrics,
            **rank_metrics
        })

    # Print final comparative results table
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL MODEL CONFIGURATION COMPARATIVE RESULTS")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)

    # Save to CSV and Markdown in workspace
    df_results.to_csv("metrics.csv", index=False)
    with open("metrics.md", "w", encoding="utf-8") as f:
        f.write("# SQuAD Sentence Salience Comparative Results\n\n")
        f.write(df_results.to_markdown(index=False))
    print("Saved comparative metrics to metrics.csv and metrics.md in workspace.")

    # Save copies to brain artifacts directory
    brain_dir = r"C:\Users\maddu\.gemini\antigravity\brain\64294085-ec70-478d-8d81-2ec235975552"
    if os.path.exists(brain_dir):
        df_results.to_csv(os.path.join(brain_dir, "metrics.csv"), index=False)
        with open(os.path.join(brain_dir, "metrics.md"), "w", encoding="utf-8") as f:
            f.write("# SQuAD Sentence Salience Comparative Results\n\n")
            f.write(df_results.to_markdown(index=False))
        print("Saved copies to brain artifacts directory.")

if __name__ == "__main__":
    main()
