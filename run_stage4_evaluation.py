import os
import time
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.classifiers.rule_based_rst import RuleBasedRSTClassifier
from src.classifiers.logistic_reg import TabularClassifierWrapper
from src.classifiers.hybrid_bert import HybridBERTClassifier
from src.classifiers.lgsm import LGSMSaliencyClassifier

# Re-use metric computation functions from Stage 3 for consistency
def compute_classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

def compute_ranking_metrics(records, probas):
    df_eval = pd.DataFrame({
        "question_id": [r["question_id"] for r in records],
        "label": [r["binary_label"] for r in records],
        "prob": probas
    })
    
    mrr_list = []
    map_list = []
    ndcg_list = []
    
    for q_id, group in df_eval.groupby("question_id"):
        sorted_group = group.sort_values(by="prob", ascending=False).reset_index(drop=True)
        
        # MRR
        pos_indices = sorted_group.index[sorted_group["label"] == 1].tolist()
        if pos_indices:
            mrr_list.append(1.0 / (pos_indices[0] + 1))
        else:
            mrr_list.append(0.0)
            
        # MAP
        hit_ranks = []
        num_hits = 0
        sum_precisions = 0.0
        for i, row in sorted_group.iterrows():
            if row["label"] == 1:
                num_hits += 1
                sum_precisions += num_hits / (i + 1)
        if num_hits > 0:
            map_list.append(sum_precisions / num_hits)
        else:
            map_list.append(0.0)
            
        # NDCG
        dcg = 0.0
        for i, row in sorted_group.iterrows():
            if row["label"] == 1:
                dcg += 1.0 / np.log2(i + 2)
        # IDCG (ideal is having all positive labels at the top)
        idcg = sum(1.0 / np.log2(j + 2) for j in range(int(group["label"].sum())))
        if idcg > 0:
            ndcg_list.append(dcg / idcg)
        else:
            ndcg_list.append(0.0)
            
    return {
        "MRR": np.mean(mrr_list),
        "MAP": np.mean(map_list),
        "NDCG": np.mean(ndcg_list)
    }

def compute_macro_metrics(records, preds):
    df_eval = pd.DataFrame({
        "question_id": [r["question_id"] for r in records],
        "label": [r["binary_label"] for r in records],
        "pred": preds
    })
    
    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    
    for q_id, group in df_eval.groupby("question_id"):
        y_true_g = group["label"].values
        y_pred_g = group["pred"].values
        
        acc_list.append(accuracy_score(y_true_g, y_pred_g))
        prec_list.append(precision_score(y_true_g, y_pred_g, zero_division=0))
        rec_list.append(recall_score(y_true_g, y_pred_g, zero_division=0))
        f1_list.append(f1_score(y_true_g, y_pred_g, zero_division=0))
        
    return {
        "Macro_Accuracy": np.mean(acc_list),
        "Macro_Precision": np.mean(prec_list),
        "Macro_Recall": np.mean(rec_list),
        "Macro_F1": np.mean(f1_list)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.35, help="Classification probability threshold")
    parser.add_argument("--cache_path", type=str, default="features_cache.pkl", help="Cached features path")
    args = parser.parse_args()
    
    print("="*80)
    print(f"STAGE 4: STANDALONE EVALUATION RUNNER (THRESHOLD: {args.threshold})")
    print("="*80)
    
    if not os.path.exists(args.cache_path):
        print(f"Error: Cache file '{args.cache_path}' not found.")
        return
        
    print(f"Loading cached validation features from '{args.cache_path}'...")
    with open(args.cache_path, "rb") as f:
        cache_data = pickle.load(f)
    train_records = cache_data["train"]
    val_records = cache_data["validation"]
    print(f"Loaded {len(train_records)} train records and {len(val_records)} validation records.")
    
    checkpoint_dir = "checkpoints"
    results = []
    
    splits = [
        ("train", train_records),
        ("validation", val_records)
    ]
    
    for split_name, split_records in splits:
        print(f"\n[{time.strftime('%X')}] Evaluating on '{split_name}' split...")
        y_true = np.array([r["binary_label"] for r in split_records])
        
        # -------------------------------------------------------------------------
        # CONFIG 1: RST Rule-Based Heuristics
        # -------------------------------------------------------------------------
        print(f"Evaluating Config 1: RST Rule-Based ({split_name})...")
        rb_classifier = RuleBasedRSTClassifier()
        rst_feats = [r["features"] for r in split_records]
        probas = rb_classifier.predict_proba(rst_feats)
        preds = (probas >= args.threshold).astype(int)
        
        cls_metrics = compute_classification_metrics(y_true, preds)
        rank_metrics = compute_ranking_metrics(split_records, probas)
        macro_metrics = compute_macro_metrics(split_records, preds)
        
        results.append({
            "Model Configuration": "1. RST Rule-Based",
            "Balancing": "None",
            "Split": split_name,
            **cls_metrics,
            **macro_metrics,
            **rank_metrics
        })

        # -------------------------------------------------------------------------
        # CONFIG 2-5: Logistic Regression Configurations
        # -------------------------------------------------------------------------
        balancing_methods = ["None", "Pairwise", "Cluster", "RST-Neighborhood", "DSNB"]
        modes = ["rst", "linguistic", "surprisal", "combined"]
        
        for balancing in balancing_methods:
            for mode in modes:
                config_num = {"rst": 2, "linguistic": 3, "surprisal": 4, "combined": 5}[mode]
                disp_name = mode.upper() if mode == "rst" else mode.replace('_', ' ').title()
                model_name = f"{config_num}. LR ({disp_name})"
                
                checkpoint_path = os.path.join(checkpoint_dir, f"lr_{mode}_{balancing}.joblib")
                if not os.path.exists(checkpoint_path):
                    continue
                    
                lr_wrapper = TabularClassifierWrapper(feature_mode=mode, use_soft_targets=False)
                lr_wrapper.load(checkpoint_path)
                
                probas = lr_wrapper.predict_proba(split_records)
                preds = (probas >= args.threshold).astype(int)
                
                cls_metrics = compute_classification_metrics(y_true, preds)
                rank_metrics = compute_ranking_metrics(split_records, probas)
                macro_metrics = compute_macro_metrics(split_records, preds)
                
                results.append({
                    "Model Configuration": model_name,
                    "Balancing": balancing,
                    "Split": split_name,
                    **cls_metrics,
                    **macro_metrics,
                    **rank_metrics
                })

        # -------------------------------------------------------------------------
        # CONFIG 6-9, 11: BERT-Based Configurations
        # -------------------------------------------------------------------------
        bert_configs = [
            ("gated_all", "6. Gated BERT (Context Features)"),
            ("film_rst_skip", "7. FiLM BERT (RST Modulation)"),
            ("concat_all", "8. Concat BERT (Context Features)"),
            ("no_rst", "9. Gated BERT (No RST)"),
            ("heuristic_guided_rst", "11. Heuristic-Guided BERT (RST Prior)")
        ]
        
        for balancing in balancing_methods:
            for mode, name in bert_configs:
                checkpoint_path = os.path.join(checkpoint_dir, f"bert_{mode}_{balancing}.pt")
                if not os.path.exists(checkpoint_path):
                    continue
                    
                print(f"Loading & evaluating {name} ({balancing} balancing) on {split_name}...")
                bert_classifier = HybridBERTClassifier(mode=mode, device="cuda")
                bert_classifier.load(checkpoint_path, device="cuda")
                
                probas = bert_classifier.predict_proba(split_records, batch_size=32)
                preds = (probas >= args.threshold).astype(int)
                
                cls_metrics = compute_classification_metrics(y_true, preds)
                rank_metrics = compute_ranking_metrics(split_records, probas)
                macro_metrics = compute_macro_metrics(split_records, preds)
                
                results.append({
                    "Model Configuration": name,
                    "Balancing": balancing,
                    "Split": split_name,
                    **cls_metrics,
                    **macro_metrics,
                    **rank_metrics
                })

        # -------------------------------------------------------------------------
        # CONFIG 12: LGSM Model
        # -------------------------------------------------------------------------
        checkpoint_path = os.path.join(checkpoint_dir, "lgsm.pt")
        if os.path.exists(checkpoint_path):
            print(f"Loading & evaluating Config 12: LGSM Saliency Classifier on {split_name}...")
            lgsm_classifier = LGSMSaliencyClassifier(pretrained_name="bert-base-uncased", device="cuda")
            lgsm_classifier.load(checkpoint_path, device="cuda")
            
            probas, _ = lgsm_classifier.predict_proba(split_records, batch_size=4)
            preds = (probas >= args.threshold).astype(int)
            
            cls_metrics = compute_classification_metrics(y_true, preds)
            rank_metrics = compute_ranking_metrics(split_records, probas)
            macro_metrics = compute_macro_metrics(split_records, preds)
            
            results.append({
                "Model Configuration": "12. LGSM",
                "Balancing": "None",
                "Split": split_name,
                **cls_metrics,
                **macro_metrics,
                **rank_metrics
            })

    # Compile and Sort Results
    df_results = pd.DataFrame(results)
    balancing_order = ["None", "Pairwise", "Cluster", "RST-Neighborhood", "DSNB"]
    df_results["Balancing"] = df_results["Balancing"].fillna("None").astype(str)
    df_results["Balancing"] = pd.Categorical(df_results["Balancing"], categories=balancing_order, ordered=True)
    
    # Add Split ordering
    split_order = ["train", "validation"]
    df_results["Split"] = pd.Categorical(df_results["Split"], categories=split_order, ordered=True)
    
    model_order = [
        "1. RST Rule-Based",
        "2. LR (RST)",
        "3. LR (Linguistic)",
        "4. LR (Surprisal)",
        "5. LR (Combined)",
        "6. Gated BERT (Context Features)",
        "7. FiLM BERT (RST Modulation)",
        "8. Concat BERT (Context Features)",
        "9. Gated BERT (No RST)",
        "11. Heuristic-Guided BERT (RST Prior)",
        "12. LGSM"
    ]
    df_results["Model Configuration"] = pd.Categorical(df_results["Model Configuration"], categories=model_order, ordered=True)
    
    # Sort by Configuration, Balancing, and Split
    df_results = df_results.sort_values(by=["Model Configuration", "Balancing", "Split"]).reset_index(drop=True)
    
    df_results["Balancing"] = df_results["Balancing"].astype(str)
    df_results["Model Configuration"] = df_results["Model Configuration"].astype(str)
    df_results["Split"] = df_results["Split"].astype(str)
    
    print("\n" + "="*80)
    print("STAGE 4 RESULTS (THRESHOLD = {})".format(args.threshold))
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    # Save outputs
    df_results.to_csv("metrics.csv", index=False)
    with open("metrics.md", "w", encoding="utf-8") as f:
        f.write("# SQuAD Sentence Salience Comparative Results (Stage 4, Threshold: {}, Train vs Val)\n\n".format(args.threshold))
        f.write(df_results.to_markdown(index=False))
    print(f"[{time.strftime('%X')}] Saved comparative metrics to metrics.csv and metrics.md in workspace.")

    # Saved to workspace
        
    print("="*80)
    print("STAGE 4 EVALUATION RUNNER COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
