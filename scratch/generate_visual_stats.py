import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    cache_path = "features_cache_deletion.pkl"
    workspace_img_dir = "docs/images"
    brain_dir = r"C:\Users\maddu\.gemini\antigravity\brain\64294085-ec70-478d-8d81-2ec235975552"
    
    if not os.path.exists(cache_path):
        # Fallback if deletion cache is not present
        cache_path = "features_cache.pkl"
        
    if not os.path.exists(cache_path):
        print(f"Error: Cache file '{cache_path}' not found.")
        return
        
    os.makedirs(workspace_img_dir, exist_ok=True)
    
    print("Loading cached features...")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
        
    # Combine train and validation splits for overall dataset statistics
    all_records = cache_data["train"] + cache_data["validation"]
    
    # Flatten records to DataFrame
    flat_records = []
    for r in all_records:
        rec = {
            "question_id": r["question_id"],
            "sentence_idx": r["sentence_idx"],
            "binary_label": r["binary_label"],
            "sentence_text": r["sentence_text"]
        }
        for k, v in r["features"].items():
            rec[k] = v
        flat_records.append(rec)
        
    df = pd.DataFrame(flat_records)
    print(f"DataFrame loaded with {len(df)} rows and {len(df.columns)} columns.")
    
    # -------------------------------------------------------------
    # Styling Setup for Matplotlib (Modern, Clean Dark Theme Style)
    # -------------------------------------------------------------
    plt.rcParams['figure.facecolor'] = '#121212'
    plt.rcParams['axes.facecolor'] = '#1e1e1e'
    plt.rcParams['text.color'] = '#e0e0e0'
    plt.rcParams['axes.labelcolor'] = '#e0e0e0'
    plt.rcParams['xtick.color'] = '#a0a0a0'
    plt.rcParams['ytick.color'] = '#a0a0a0'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['font.size'] = 11
    
    # Color palette
    color_primary = '#3b82f6'    # Vibrant Blue
    color_secondary = '#10b981'  # Teal
    color_accent = '#f59e0b'     # Amber/Orange
    color_dark_grey = '#4b5563'
    
    # =========================================================================
    # PLOT 1: Positional Bias of Salient Sentences
    # =========================================================================
    print("Generating Plot 1: Positional Bias...")
    pos_counts = df[df["binary_label"] == 1]["sentence_idx"].value_counts().sort_index()
    total_salient = len(df[df["binary_label"] == 1])
    pos_pcts = (pos_counts / total_salient) * 100
    
    # Keep only indices 0 to 7, and group the rest into "8+"
    indices_filtered = list(range(8))
    counts_filtered = [pos_pcts.get(i, 0.0) for i in indices_filtered]
    
    # Calculate index 8+
    rest_pct = sum(v for k, v in pos_pcts.items() if k >= 8)
    indices_filtered.append("8+")
    counts_filtered.append(rest_pct)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(i) for i in indices_filtered], counts_filtered, color=color_primary, width=0.6, edgecolor='#2563eb', linewidth=1)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#ffffff')
                    
    ax.set_title("Distribution of Salient Sentences by Linear Position", pad=20, weight='bold', fontsize=14)
    ax.set_xlabel("Sentence Index in Context Paragraph", labelpad=12)
    ax.set_ylabel("Percentage of Answer Sentences (%)", labelpad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.1)
    plt.tight_layout()
    
    plot1_workspace = os.path.join(workspace_img_dir, "positional_bias.png")
    plt.savefig(plot1_workspace, dpi=150, facecolor='#121212')
    if os.path.exists(brain_dir):
        plt.savefig(os.path.join(brain_dir, "positional_bias.png"), dpi=150, facecolor='#121212')
    plt.close()
    
    # =========================================================================
    # PLOT 2: Feature Correlations
    # =========================================================================
    print("Generating Plot 2: Feature Correlations...")
    features_to_check = {
        "align_sem_sim": "SBERT Cosine Similarity",
        "align_jaccard": "Lemma Jaccard Overlap",
        "align_rouge_l_recall": "ROUGE-L LCS Recall",
        "rel_rst_n_ratio": "Relative RST Nucleus Ratio",
        "word_count": "Sentence Word Count",
        "char_count": "Sentence Character Count",
        "surp_deletion_drop": "GPT-2 Deletion Coherence Drop",
        "surp_mean": "Mean GPT-2 Surprisal",
        "rst_mean_depth": "Mean RST Hierarchical Depth"
    }
    
    corrs = []
    labels = []
    colors = []
    
    for feat, label in features_to_check.items():
        if feat in df.columns:
            val = df["binary_label"].corr(df[feat])
            corrs.append(val)
            labels.append(label)
            # Assign color based on positive or negative correlation
            colors.append(color_secondary if val >= 0 else '#ef4444')
            
    # Sort by correlation value
    sorted_idx = np.argsort(corrs)
    corrs_sorted = [corrs[i] for i in sorted_idx]
    labels_sorted = [labels[i] for i in sorted_idx]
    colors_sorted = [colors[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(labels_sorted, corrs_sorted, color=colors_sorted, height=0.6)
    
    # Add values at the end of the bars
    for bar in bars:
        width = bar.get_width()
        align = 'left' if width < 0 else 'right'
        offset = -25 if width < 0 else 5
        color_text = '#ffffff'
        
        ax.annotate(f'{width:+.3f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(offset, 0),
                    textcoords="offset points",
                    ha=align, va='center', fontsize=9, color=color_text, fontweight='bold')
                    
    ax.axvline(0, color='#555555', linewidth=1, linestyle='-')
    ax.set_title("Pearson Correlation ($r$) of Extracted Features with Salience", pad=20, weight='bold', fontsize=14)
    ax.set_xlabel("Pearson Correlation Coefficient", labelpad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.1)
    plt.tight_layout()
    
    plot2_workspace = os.path.join(workspace_img_dir, "feature_correlations.png")
    plt.savefig(plot2_workspace, dpi=150, facecolor='#121212')
    if os.path.exists(brain_dir):
        plt.savefig(os.path.join(brain_dir, "feature_correlations.png"), dpi=150, facecolor='#121212')
    plt.close()
    
    # =========================================================================
    # PLOT 3: RST Relation Frequency Comparison (Salient vs. Non-Salient)
    # =========================================================================
    print("Generating Plot 3: RST Relation Comparison...")
    rst_relations = [
        "rst_rel_elaboration_count",
        "rst_rel_attribution_count",
        "rst_rel_background_count",
        "rst_rel_cause_count",
        "rst_rel_result_count",
        "rst_rel_contrast_count",
        "rst_rel_joint_count"
    ]
    
    relation_labels = [r.replace("rst_rel_", "").replace("_count", "").capitalize() for r in rst_relations]
    
    # Calculate means
    means_salient = []
    means_non_salient = []
    
    df_salient = df[df["binary_label"] == 1]
    df_non_salient = df[df["binary_label"] == 0]
    
    for rel in rst_relations:
        if rel in df.columns:
            means_salient.append(df_salient[rel].mean())
            means_non_salient.append(df_non_salient[rel].mean())
        else:
            means_salient.append(0.0)
            means_non_salient.append(0.0)
            
    x = np.arange(len(relation_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, means_salient, width, label='Salient (Answer)', color=color_secondary, edgecolor='#059669')
    rects2 = ax.bar(x + width/2, means_non_salient, width, label='Non-Salient (Noise)', color=color_dark_grey, edgecolor='#374151')
    
    ax.set_ylabel('Mean Relation Count per Sentence', labelpad=12)
    ax.set_title('Rhetorical Structure Theory (RST) Relations Frequencies', pad=20, weight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(relation_labels)
    ax.legend(frameon=True, facecolor='#1e1e1e', edgecolor='#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.1)
    plt.tight_layout()
    
    plot3_workspace = os.path.join(workspace_img_dir, "rst_relations_comparison.png")
    plt.savefig(plot3_workspace, dpi=150, facecolor='#121212')
    if os.path.exists(brain_dir):
        plt.savefig(os.path.join(brain_dir, "rst_relations_comparison.png"), dpi=150, facecolor='#121212')
    plt.close()

    # =========================================================================
    # PLOT 4: Length Statistics Comparison
    # =========================================================================
    print("Generating Plot 4: Length Comparison...")
    mean_words_salient = df_salient["word_count"].mean()
    mean_words_non_salient = df_non_salient["word_count"].mean()
    mean_chars_salient = df_salient["char_count"].mean()
    mean_chars_non_salient = df_non_salient["char_count"].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Word count bar
    ax1.bar(["Salient", "Non-Salient"], [mean_words_salient, mean_words_non_salient], color=[color_accent, color_dark_grey], width=0.5)
    ax1.set_title("Average Word Count", fontsize=12, weight='bold')
    ax1.set_ylabel("Words")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # Add labels
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height() - 2.5),
                    ha='center', va='center', xytext=(0, 0), textcoords='offset points', color='#ffffff', fontweight='bold')
                    
    # Character count bar
    ax2.bar(["Salient", "Non-Salient"], [mean_chars_salient, mean_chars_non_salient], color=[color_accent, color_dark_grey], width=0.5)
    ax2.set_title("Average Character Count", fontsize=12, weight='bold')
    ax2.set_ylabel("Characters")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # Add labels
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height() - 15),
                    ha='center', va='center', xytext=(0, 0), textcoords='offset points', color='#ffffff', fontweight='bold')
                    
    fig.suptitle("Sentence Length Comparison: Salient vs. Non-Salient", fontsize=14, weight='bold', y=0.98)
    plt.tight_layout()
    
    plot4_workspace = os.path.join(workspace_img_dir, "length_comparison.png")
    plt.savefig(plot4_workspace, dpi=150, facecolor='#121212')
    if os.path.exists(brain_dir):
        plt.savefig(os.path.join(brain_dir, "length_comparison.png"), dpi=150, facecolor='#121212')
    plt.close()
    
    print("All plots generated and saved successfully!")

if __name__ == "__main__":
    main()
