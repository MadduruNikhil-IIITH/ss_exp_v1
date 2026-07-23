import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def main():
    print("="*80)
    print("INTERPRETABILITY STUDY: LGSM GATE BEHAVIOR ANALYSIS")
    print("="*80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_path = os.path.abspath(os.path.join(script_dir, "../lgsm_predictions.pkl"))
    
    if not os.path.exists(pred_path):
        print(f"Error: LGSM predictions file '{pred_path}' not found. Run Stage 3 first!")
        return

    print(f"Loading LGSM validation predictions from '{pred_path}'...")
    with open(pred_path, "rb") as f:
        data = pickle.load(f)

    val_records = data["val_records"]
    y_val_true = data["y_val_true"]
    probas = data["probas"]
    gates = data["gates"]

    print(f"Loaded predictions for {len(val_records)} records.")

    # Create DataFrame for analysis
    df = pd.DataFrame({
        "question_id": [r["question_id"] for r in val_records],
        "sentence_idx": [int(r["sentence_idx"]) for r in val_records],
        "label": y_val_true,
        "prob": probas,
        "gate": gates
    })

    # 1. Gate value distribution
    mean_gate = df["gate"].mean()
    std_gate = df["gate"].std()
    median_gate = df["gate"].median()
    print(f"\nGate Value Distribution:")
    print(f"  - Mean:   {mean_gate:.4f}")
    print(f"  - Median: {median_gate:.4f}")
    print(f"  - Std:    {std_gate:.4f}")

    # 2. Gate vs sentence_idx (gating arc)
    idx_groups = df.groupby("sentence_idx")["gate"].agg(["mean", "std", "count"]).reset_index()
    # Filter positions with at least 5 instances
    idx_groups = idx_groups[idx_groups["count"] >= 5].reset_index(drop=True)
    print("\nGate Value by Sentence Index:")
    print(idx_groups.to_string(index=False))

    # 3. Gate vs Salience Prediction Correlation
    corr, p_val = pearsonr(df["gate"], df["prob"])
    print(f"\nCorrelation between gate alpha_t and saliency probability: {corr:.4f} (p-value = {p_val:.4e})")

    corr_sal, _ = pearsonr(df[df["label"] == 1]["gate"], df[df["label"] == 1]["prob"]) if len(df[df["label"] == 1]) > 1 else (0, 0)
    corr_nsal, _ = pearsonr(df[df["label"] == 0]["gate"], df[df["label"] == 0]["prob"]) if len(df[df["label"] == 0]) > 1 else (0, 0)
    print(f"  - Correlation in Salient class:    {corr_sal:.4f}")
    print(f"  - Correlation in Non-Salient class: {corr_nsal:.4f}")

    # Generate Gating Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Gate Histogram
    axes[0].hist(df["gate"], bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=mean_gate, color='#e74c3c', linestyle='--', label=f'Mean ({mean_gate:.2f})')
    axes[0].set_title('Distribution of Gate Values (alpha_t)')
    axes[0].set_xlabel('Gate Value alpha_t')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Plot 2: Gate value over Sentence index (gating progression)
    axes[1].errorbar(idx_groups["sentence_idx"], idx_groups["mean"], yerr=idx_groups["std"], fmt='-o', color='#3498db', ecolor='#95a5a6', capsize=4, elinewidth=1, linewidth=2)
    axes[1].set_title('Gate Value over Sentence Position')
    axes[1].set_xlabel('Sentence Index')
    axes[1].set_ylabel('Mean Gate Value alpha_t (with std)')
    axes[1].set_xticks(idx_groups["sentence_idx"])
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # Plot 3: Scatter Plot of Gate value vs Prediction Probability
    axes[2].scatter(df["gate"], df["prob"], alpha=0.4, color='#9b59b6', s=15)
    # Fit regression line
    m, b = np.polyfit(df["gate"], df["prob"], 1)
    axes[2].plot(df["gate"], m*df["gate"] + b, color='#e74c3c', linewidth=2, label=f'r = {corr:.2f}')
    axes[2].set_title('Gate Value vs. Salience Probability')
    axes[2].set_xlabel('Gate Value alpha_t')
    axes[2].set_ylabel('Saliency Probability')
    axes[2].legend()
    axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    docs_images_dir = os.path.abspath(os.path.join(script_dir, "../docs/images"))
    os.makedirs(docs_images_dir, exist_ok=True)
    
    plot_path = os.path.join(docs_images_dir, "gating_progression.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved gating analysis plots to '{plot_path}'")
    
    # Write Markdown report
    docs_dir = os.path.abspath(os.path.join(script_dir, "../docs"))
    os.makedirs(docs_dir, exist_ok=True)
    report_path = os.path.join(docs_dir, "gating_analysis_report.md")
    
    def write_report(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# SQuAD LGSM Gating Behavior Analysis Report\n\n")
            f.write("This report analyzes the behavior of the scalar gating parameter $\\alpha_t \\in (0, 1)$ in the Linguistically-Grounded Saliency Model (LGSM) on SQuAD validation passages.\n\n")
            
            f.write("## 1. Gating Statistics Summary\n")
            f.write(f"- **Mean Gate Value**: `{mean_gate:.4f}`\n")
            f.write(f"- **Median Gate Value**: `{median_gate:.4f}`\n")
            f.write(f"- **Standard Deviation**: `{std_gate:.4f}`\n")
            f.write(f"- **Linear Correlation ($r$)**: `{corr:.4f}` (p-value = `{p_val:.4e}`)\n\n")
            
            f.write("### Position-Wise Gate Values\n")
            f.write("| Sentence Index | Mean Gate Value (alpha) | Std Dev | Sample Count |\n")
            f.write("| --- | --- | --- | --- |\n")
            for _, r in idx_groups.iterrows():
                f.write(f"| Sentence {int(r['sentence_idx'])} | `{r['mean']:.4f}` | `{r['std']:.4f}` | {int(r['count'])} |\n")
            f.write("\n")
            
            f.write("## 2. Key Findings & Discussion\n\n")
            
            # Interpret gate values
            f.write("### A. Stream Weight Distribution\n")
            if mean_gate > 0.6:
                f.write(f"The mean gate value of `{mean_gate:.4f}` indicates that the model relies **more heavily on the linguistic stream** (RST, surprisal, syntax) than on BERT semantic text embeddings.\n")
            elif mean_gate < 0.4:
                f.write(f"The mean gate value of `{mean_gate:.4f}` indicates that the model relies **more heavily on the semantic stream** (BERT representations) than on explicit features.\n")
            else:
                f.write(f"The mean gate value of `{mean_gate:.4f}` indicates that the model maintains a **balanced contribution** (~50% semantics, ~50% structure) across the validation passages.\n")
            f.write("\n")
            
            # Position-wise trend
            f.write("### B. Positional Gating Progression (Arc)\n")
            means = idx_groups["mean"].values
            if len(means) > 2:
                slope = (means[-1] - means[0]) / len(means)
                f.write(f"The gate value progresses with a slope of `{slope:.4f}` across sentence indices. ")
                if abs(slope) < 0.01:
                    f.write("This flat temporal trend indicates that linguistic features are continuously and evenly integrated throughout the passage, rather than isolated to specific positions.\n")
                elif slope > 0:
                    f.write("The rising gate value indicates that the model relies more on structural and discourse features for later sentences in the passage.\n")
                else:
                    f.write("The declining gate value indicates that the model relies more on text semantics for later sentences in the passage.\n")
            f.write("\n")
            
            # Gating-Prediction correlation
            f.write("### C. Gating-Prediction Correlation (Negative Filter Hypothesis)\n")
            if corr < -0.3:
                f.write(f"The strong negative correlation of `{corr:.4f}` indicates that higher reliance on linguistic features predicts **non-saliency**. This strongly supports the **Negative Filter Hypothesis** from the movie screenplay paper, suggesting that the model uses structural features (such as Satellite relation counts) as precision filters to confidently reject background sentences.\n")
            elif corr > 0.3:
                f.write(f"The positive correlation of `{corr:.4f}` indicates that higher reliance on linguistic features directly predicts **sentence salience**.\n")
            else:
                f.write(f"The low correlation of `{corr:.4f}` suggests that the gating coefficient adjusts dynamically and relationally to select salient sentences, rather than acting as a simple monotonic filter.\n")

    write_report(report_path)
    print(f"Gating Analysis completed successfully. Reports saved to '{report_path}'.")

if __name__ == "__main__":
    main()
