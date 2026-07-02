import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

from src.classifiers.rule_based_rst import RuleBasedRSTClassifier
from src.classifiers.logistic_reg import TabularClassifierWrapper
from src.classifiers.hybrid_bert import HybridBERTClassifier
from src.classifiers.lgsm import LGSMSaliencyClassifier

def compute_top_k_recall(records, probas, k):
    df_eval = pd.DataFrame({
        "question_id": [r["question_id"] for r in records],
        "label": [r["binary_label"] for r in records],
        "prob": probas
    })
    
    hits = 0
    total = 0
    
    for q_id, group in df_eval.groupby("question_id"):
        sorted_group = group.sort_values(by="prob", ascending=False).reset_index(drop=True)
        # Find index of the true salient sentence (there is exactly 1 per question_id)
        pos_indices = sorted_group.index[sorted_group["label"] == 1].tolist()
        if pos_indices:
            # If the rank is within top-k (0-indexed, so < k)
            if pos_indices[0] < k:
                hits += 1
        total += 1
        
    return hits / total if total > 0 else 0.0

def main():
    print("="*80)
    print("STAGE 4: DIAGNOSTICS & PLOTTING PIPELINE")
    print("="*80)

    cache_path = "features_cache.pkl"
    if not os.path.exists(cache_path):
        print(f"Error: Cache file '{cache_path}' not found.")
        return
        
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
    val_records = cache_data["validation"]
    y_val_true = np.array([r["binary_label"] for r in val_records])
    
    checkpoint_dir = "checkpoints"
    os.makedirs(os.path.join("docs", "images"), exist_ok=True)
    
    # 1. Load the target models
    print("Loading models for diagnostics...")
    
    # Model A: Config 1 RST Rule-Based
    rb_classifier = RuleBasedRSTClassifier()
    val_rst_feats = [r["features"] for r in val_records]
    probs_rb = rb_classifier.predict_proba(val_rst_feats)
    
    # Model B: Config 5 LR Combined (DSNB)
    probs_lr = np.zeros_like(y_val_true, dtype=float)
    lr_path = os.path.join(checkpoint_dir, "lr_combined_DSNB.joblib")
    if os.path.exists(lr_path):
        lr_wrapper = TabularClassifierWrapper(feature_mode="combined", use_soft_targets=False)
        lr_wrapper.load(lr_path)
        probs_lr = lr_wrapper.predict_proba(val_records)
    else:
        print("Warning: lr_combined_DSNB.joblib checkpoint not found.")
        
    # Model C: Config 6 Gated BERT (Pairwise)
    probs_gated = np.zeros_like(y_val_true, dtype=float)
    bert_path = os.path.join(checkpoint_dir, "bert_gated_all_Pairwise.pt")
    if os.path.exists(bert_path):
        bert_classifier = HybridBERTClassifier(mode="gated_all", device="cuda")
        bert_classifier.load(bert_path, device="cuda")
        probs_gated = bert_classifier.predict_proba(val_records, batch_size=32)
    else:
        print("Warning: bert_gated_all_Pairwise.pt checkpoint not found.")
        
    # Model D: Config 12 LGSM (None)
    probs_lgsm = np.zeros_like(y_val_true, dtype=float)
    lgsm_path = os.path.join(checkpoint_dir, "lgsm.pt")
    if os.path.exists(lgsm_path):
        lgsm_classifier = LGSMSaliencyClassifier(pretrained_name="bert-base-uncased", device="cuda")
        lgsm_classifier.load(lgsm_path, device="cuda")
        probs_lgsm, _ = lgsm_classifier.predict_proba(val_records, batch_size=4)
    else:
        print("Warning: lgsm.pt checkpoint not found.")

    # -------------------------------------------------------------------------
    # PART 1: Threshold Sensitivity Sweep
    # -------------------------------------------------------------------------
    print("\nRunning Threshold Sensitivity Sweep (0.0 to 1.0)...")
    thresholds = np.linspace(0.0, 1.0, 21)
    
    f1_rb, f1_lr, f1_gated, f1_lgsm = [], [], [], []
    prec_lgsm, rec_lgsm = [], []
    
    for t in thresholds:
        # F1 sweeps
        f1_rb.append(f1_score(y_val_true, (probs_rb >= t).astype(int), zero_division=0))
        f1_lr.append(f1_score(y_val_true, (probs_lr >= t).astype(int), zero_division=0))
        f1_gated.append(f1_score(y_val_true, (probs_gated >= t).astype(int), zero_division=0))
        f1_lgsm.append(f1_score(y_val_true, (probs_lgsm >= t).astype(int), zero_division=0))
        
        # LGSM precision/recall trade-off
        prec_lgsm.append(precision_score(y_val_true, (probs_lgsm >= t).astype(int), zero_division=0))
        rec_lgsm.append(recall_score(y_val_true, (probs_lgsm >= t).astype(int), zero_division=0))

    # Plot 1: F1 Curves for all models
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, f1_rb, label="RST Rule-Based", color="#95a5a6", linestyle="--", linewidth=1.5)
    plt.plot(thresholds, f1_lr, label="LR Combined (DSNB)", color="#f1c40f", linewidth=2)
    plt.plot(thresholds, f1_gated, label="Gated BERT (Pairwise)", color="#e74c3c", linewidth=2)
    plt.plot(thresholds, f1_lgsm, label="LGSM (None)", color="#3498db", linewidth=2.5)
    plt.axvline(x=0.35, color="black", linestyle=":", label="Proposed Threshold (0.35)")
    plt.title("F1-Score Sensitivity across Thresholds")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Micro-F1 Score")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Plot 2: LGSM Precision-Recall Curves
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, prec_lgsm, label="Precision", color="#2ecc71", linewidth=2)
    plt.plot(thresholds, rec_lgsm, label="Recall", color="#e67e22", linewidth=2)
    plt.plot(thresholds, f1_lgsm, label="F1-Score", color="#3498db", linewidth=2)
    plt.axvline(x=0.35, color="black", linestyle=":")
    plt.title("LGSM Threshold Sensitivity Trade-Off")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Metric Score")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    t_plot_path = os.path.join("docs", "images", "threshold_sensitivity.png")
    plt.savefig(t_plot_path, dpi=150)
    print(f"Saved threshold sensitivity plot to '{t_plot_path}'")
    
    # Saved to workspace
        
    # Save raw sweep metrics to CSV
    sweep_df = pd.DataFrame({
        "threshold": thresholds,
        "rst_rule_based_f1": f1_rb,
        "lr_combined_dsnb_f1": f1_lr,
        "gated_bert_pairwise_f1": f1_gated,
        "lgsm_f1": f1_lgsm,
        "lgsm_precision": prec_lgsm,
        "lgsm_recall": rec_lgsm
    })
    csv_workspace_path = os.path.join("docs", "threshold_sweep_results.csv")
    sweep_df.to_csv(csv_workspace_path, index=False)
    print(f"Saved threshold sweep metrics to '{csv_workspace_path}'")
    # Saved to workspace

    # -------------------------------------------------------------------------
    # PART 2: Top-K Recall Sweep
    # -------------------------------------------------------------------------
    print("\nRunning Top-K Recall Sweep (K = 1 to 5)...")
    k_vals = [1, 2, 3, 4, 5]
    
    rec_k_rb = [compute_top_k_recall(val_records, probs_rb, k) for k in k_vals]
    rec_k_lr = [compute_top_k_recall(val_records, probs_lr, k) for k in k_vals]
    rec_k_gated = [compute_top_k_recall(val_records, probs_gated, k) for k in k_vals]
    rec_k_lgsm = [compute_top_k_recall(val_records, probs_lgsm, k) for k in k_vals]
    
    plt.figure(figsize=(7, 5))
    plt.plot(k_vals, rec_k_rb, label="RST Rule-Based", color="#95a5a6", linestyle="--", marker="o")
    plt.plot(k_vals, rec_k_lr, label="LR Combined (DSNB)", color="#f1c40f", marker="s")
    plt.plot(k_vals, rec_k_gated, label="Gated BERT (Pairwise)", color="#e74c3c", marker="^")
    plt.plot(k_vals, rec_k_lgsm, label="LGSM (None)", color="#3498db", marker="D", linewidth=2.5)
    
    plt.title("Top-K Recall Comparison (Validation Passage Sentences)")
    plt.xlabel("Top-K Sentences Retained")
    plt.ylabel("Recall (Answer Retrieval Probability)")
    plt.xticks(k_vals)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    k_plot_path = os.path.join("docs", "images", "top_k_recall_curve.png")
    plt.savefig(k_plot_path, dpi=150)
    print(f"Saved Top-K recall plot to '{k_plot_path}'")
    
    # Saved to workspace
         
    print("\n" + "="*80)
    print("STAGE 4 DIAGNOSTICS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
