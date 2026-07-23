import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from src.data_processing import apply_pairwise_balancing, apply_cluster_balancing, apply_rst_balancing, apply_dsnb_balancing
from src.classifiers.rule_based_rst import RuleBasedRSTClassifier
from src.classifiers.logistic_reg import TabularClassifierWrapper
from src.classifiers.hybrid_bert import HybridBERTClassifier
from src.classifiers.lgsm import LGSMSaliencyClassifier
from src.classifiers.llm_judge import LLMJudgeClassifier
from run_stage2_feature_extraction import run_stage2_pipeline

def compute_classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

def compute_ranking_metrics(records, scores):
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
        group_sorted = sorted(group, key=lambda x: x[2], reverse=True)
        y_true_bin = [x[0] for x in group_sorted]
        y_true_soft = [x[1] for x in group_sorted]
        
        rr = 0.0
        for rank_idx, val in enumerate(y_true_bin):
            if val == 1:
                rr = 1.0 / (rank_idx + 1)
                break
        mrr_list.append(rr)
        
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
    print("="*80)
    print("STAGE 3: EXPERIMENTAL TRAINING & EVALUATION PIPELINE")
    print("="*80)
    
    cache_path = "features_cache.pkl"
    
    # Fallback to Stage 2 if cache is missing
    if not os.path.exists(cache_path):
        print(f"[{time.strftime('%X')}] Warning: Features cache '{cache_path}' not found.")
        print(f"[{time.strftime('%X')}] Automatically running Stage 2 Feature Extraction Pipeline first...")
        run_stage2_pipeline(cache_path=cache_path)
        
    print(f"[{time.strftime('%X')}] Loading cached features from '{cache_path}'...")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
        
    train_records = cache_data["train"]
    val_records = cache_data["validation"]
    
    print(f"[{time.strftime('%X')}] Loaded {len(train_records)} train records and {len(val_records)} val records.")
    
    if train_records:
        all_features = list(train_records[0]["features"].keys())
    else:
        print("Error: Empty cached records.")
        return

    y_val_true = np.array([r["binary_label"] for r in val_records])
    results = []

    # Create checkpoints directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)


    # =========================================================================
    # CONFIG 1: RST Rule-Based Heuristics
    # =========================================================================
    print(f"\n[{time.strftime('%X')}] Evaluating Config 1: RST Rule-Based Heuristics...")
    rb_classifier = RuleBasedRSTClassifier()
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

    # Pre-generate balanced datasets
    print(f"\n[{time.strftime('%X')}] Pre-generating balanced training datasets...")
    start_balance = time.time()
    balanced_datasets = {
        "None": train_records,
        "Pairwise": apply_pairwise_balancing(train_records),
        "Cluster": apply_cluster_balancing(train_records, all_features),
        "RST-Neighborhood": apply_rst_balancing(train_records),
        "DSNB": apply_dsnb_balancing(train_records)
    }
    print(f"[{time.strftime('%X')}] Balanced datasets generated in {time.time() - start_balance:.2f} seconds.")

    # =========================================================================
    # CONFIG 5: Logistic Regression (Combined)
    # =========================================================================
    balancing_methods = ["None", "Cluster", "RST-Neighborhood", "DSNB"]
    all_coefs_list = []
    
    for balancing in tqdm(balancing_methods, desc="Logistic Regression Experiments"):
        balanced_train = balanced_datasets[balancing]
        model_name = "5. LR (Combined)"
        
        lr_wrapper = TabularClassifierWrapper(feature_mode="combined", use_soft_targets=False)
        lr_wrapper.fit(balanced_train)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"lr_combined_{balancing}.joblib")
        lr_wrapper.save(checkpoint_path)
        
        # Save coefficients
        try:
            coefs = lr_wrapper.model.coef_[0]
            feature_names = lr_wrapper.selected_cols
            for feat, coef in zip(feature_names, coefs):
                all_coefs_list.append({
                    "Feature": feat,
                    "Coefficient": coef,
                    "Abs_Coefficient": np.abs(coef),
                    "Config Mode": "Combined Model (Config 5)",
                    "Balancing Mode": balancing
                })
        except Exception as e:
            print(f"\nWarning: Could not extract coefficients under {balancing}: {e}")
                
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

    # Save accumulated coefficients to lr_coefficients.csv
    if all_coefs_list:
        try:
            df_coefs = pd.DataFrame(all_coefs_list)
            df_coefs = df_coefs.sort_values(by=["Config Mode", "Balancing Mode", "Abs_Coefficient"], ascending=[True, True, False]).reset_index(drop=True)
            df_coefs.to_csv("lr_coefficients.csv", index=False)
            print(f"[{time.strftime('%X')}] Saved all LR coefficients to lr_coefficients.csv with 'Config Mode' and 'Balancing Mode' columns.")
        except Exception as e:
            print(f"\nWarning: Could not save lr_coefficients.csv: {e}")



    # =========================================================================
    # CONFIG 6 & 11: BERT-Based Configurations
    # =========================================================================
    bert_experiments = []
    for balancing in ["None", "Pairwise", "Cluster", "RST-Neighborhood", "DSNB"]:
        bert_experiments.append(("gated_all", "6. Gated BERT (Context Features)", balancing))
    
    for mode, name, balancing in tqdm(bert_experiments, desc="BERT Experiments"):
        print(f"\n[{time.strftime('%X')}] Training {name} with {balancing} balancing...")
        train_data = balanced_datasets[balancing]
        
        bert_classifier = HybridBERTClassifier(mode=mode, device="cuda")
        bert_classifier.fit(train_data, val_records, epochs=3, batch_size=32, use_soft_labels=False)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"bert_{mode}_{balancing}.pt")
        bert_classifier.save(checkpoint_path)
        
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

    # =========================================================================
    # CONFIG 12: LGSM Model (Convex Gated Fusion + Sequence Transformer + Focal Loss)
    # =========================================================================
    lgsm_experiments = [
        ("None", True),
        ("Cluster", True),
        ("RST-Neighborhood", True),
        ("DSNB", True)
    ]
    for balancing, use_rst in lgsm_experiments:
        print(f"\n[{time.strftime('%X')}] Training Config 12: LGSM Saliency Classifier with {balancing} balancing...")
        
        balanced_train = balanced_datasets[balancing]
        lgsm_classifier = LGSMSaliencyClassifier(pretrained_name="bert-base-uncased", device="cuda")
        lgsm_classifier.fit(balanced_train, val_records, epochs=4, batch_size=4, lr=2e-5, use_rst=True)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"lgsm_{balancing}.pt")
        lgsm_classifier.save(checkpoint_path)
        
        probas, gates = lgsm_classifier.predict_proba(val_records, batch_size=4)
        preds = (probas >= 0.5).astype(int)
        
        cls_metrics = compute_classification_metrics(y_val_true, preds)
        rank_metrics = compute_ranking_metrics(val_records, probas)
        
        results.append({
            "Model Configuration": "12. LGSM",
            "Balancing": balancing,
            **cls_metrics,
            **rank_metrics
        })
        
        # Save default predictions for subsequent analysis using None
        if balancing == "None":
            try:
                val_analysis_data = {
                    "val_records": val_records,
                    "y_val_true": y_val_true,
                    "probas": probas,
                    "gates": gates
                }
                with open("lgsm_predictions.pkl", "wb") as f:
                    pickle.dump(val_analysis_data, f)
                # Also save standard lgsm.pt for web app compatibility
                lgsm_classifier.save(os.path.join(checkpoint_dir, "lgsm.pt"))
                print(f"[{time.strftime('%X')}] Saved standard LGSM predictions to 'lgsm_predictions.pkl' and checkpoint to 'checkpoints/lgsm.pt'.")
            except Exception as e:
                print(f"Warning: Could not save default validation data: {e}")

    # =========================================================================
    # CONFIG 13: Zero-shot LLM Judge
    # =========================================================================
    print(f"\n[{time.strftime('%X')}] Evaluating Config 13: Zero-shot LLM Judge...")
    llm_classifier = LLMJudgeClassifier(device="cuda")
    probas_llm = llm_classifier.predict_proba(val_records)
    preds_llm = (probas_llm >= 0.5).astype(int)
    
    cls_metrics = compute_classification_metrics(y_val_true, preds_llm)
    rank_metrics = compute_ranking_metrics(val_records, probas_llm)
    
    results.append({
        "Model Configuration": "13. LLM Judge",
        "Balancing": "None",
        **cls_metrics,
        **rank_metrics
    })

    # 4. Group & Sort Comparative Metrics
    df_results = pd.DataFrame(results)
    
    balancing_order = ["None", "Pairwise", "Cluster", "RST-Neighborhood", "DSNB"]
    df_results["Balancing"] = df_results["Balancing"].fillna("None").astype(str)
    df_results["Balancing"] = pd.Categorical(df_results["Balancing"], categories=balancing_order, ordered=True)
    
    model_order = [
        "1. RST Rule-Based",
        "5. LR (Combined)",
        "6. Gated BERT (Context Features)",
        "11. Heuristic-Guided BERT (RST Prior)",
        "12. LGSM",
        "13. LLM Judge"
    ]
    df_results["Model Configuration"] = pd.Categorical(df_results["Model Configuration"], categories=model_order, ordered=True)
    df_results = df_results.sort_values(by=["Model Configuration", "Balancing"]).reset_index(drop=True)
    
    df_results["Balancing"] = df_results["Balancing"].astype(str)
    df_results["Model Configuration"] = df_results["Model Configuration"].astype(str)
    
    print("\n" + "="*80)
    print("FINAL MODEL CONFIGURATION COMPARATIVE RESULTS (GROUPED BY CLASSIFIER)")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)

    # Save to CSV and Markdown
    df_results.to_csv("metrics.csv", index=False)
    with open("metrics.md", "w", encoding="utf-8") as f:
        f.write("# SQuAD Sentence Salience Comparative Results (Grouped by Classifier)\n\n")
        f.write(df_results.to_markdown(index=False))
    print(f"[{time.strftime('%X')}] Saved comparative metrics to metrics.csv and metrics.md in workspace.")

    # Saved to workspace
        
    print("="*80)
    print("STAGE 3 PIPELINE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
