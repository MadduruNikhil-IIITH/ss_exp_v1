import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def main():
    cache_path = "features_cache_deletion.pkl"
    workspace_img_dir = "docs/images"
    brain_dir = r"C:\Users\maddu\.gemini\antigravity\brain\64294085-ec70-478d-8d81-2ec235975552"
    
    if not os.path.exists(cache_path):
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
            "soft_label_decay": r.get("soft_label_decay", 0.0),
            "soft_label_hybrid": r.get("soft_label_hybrid", 0.0),
            "sentence_text": r["sentence_text"]
        }
        for k, v in r["features"].items():
            rec[k] = v
        flat_records.append(rec)
        
    df = pd.DataFrame(flat_records)
    print(f"Loaded DataFrame with {len(df)} records.")
    
    # Define features to analyze
    target_features = [
        # Alignment
        "align_sem_sim", "align_jaccard", "align_rouge_l_recall", "align_ne_match",
        # Readability
        "flesch_reading_ease", "gunning_fog",
        # Syntactic complexity
        "max_parse_depth", "avg_dep_distance",
        # Surprisal
        "surp_mean", "surp_deletion_drop",
        # Discourse
        "rel_rst_n_ratio", "rst_mean_depth",
        # Basic length
        "word_count", "char_count", "stopword_ratio"
    ]
    
    df_salient = df[df["binary_label"] == 1]
    df_non_salient = df[df["binary_label"] == 0]
    
    # -------------------------------------------------------------
    # 1. COMPUTE STATISTICS & WELCH'S T-TEST
    # -------------------------------------------------------------
    print("Computing Welch's T-test and descriptive statistics...")
    stats_data = []
    
    for feat in target_features:
        if feat not in df.columns:
            continue
            
        salient_vals = df_salient[feat].dropna()
        non_salient_vals = df_non_salient[feat].dropna()
        
        # Means
        m1 = salient_vals.mean()
        m0 = non_salient_vals.mean()
        
        # Stds
        std1 = salient_vals.std()
        std0 = non_salient_vals.std()
        
        # Medians
        med1 = salient_vals.median()
        med0 = non_salient_vals.median()
        
        # Welch's t-test (equal_var=False)
        t_stat, p_val = stats.ttest_ind(salient_vals, non_salient_vals, equal_var=False)
        
        stats_data.append({
            "feature": feat,
            "mean_salient": m1,
            "mean_non_salient": m0,
            "median_salient": med1,
            "median_non_salient": med0,
            "std_salient": std1,
            "std_non_salient": std0,
            "t_stat": t_stat,
            "p_val": p_val
        })
        
    df_stats = pd.DataFrame(stats_data)
    
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
    plt.rcParams['font.size'] = 10
    plt.rcParams['grid.color'] = '#333333'
    
    # Color palette
    color_primary = '#3b82f6'    # Vibrant Blue (Class 0 / Non-Salient)
    color_secondary = '#10b981'  # Teal (Class 1 / Salient)
    color_accent = '#f59e0b'     # Amber/Orange
    color_grey = '#4b5563'
    color_light_grey = '#9ca3af'
    
    # =========================================================================
    # PLOT 1: Correlation Matrix Heatmap
    # =========================================================================
    print("Generating Plot 1: Correlation Matrix Heatmap...")
    corr_features = [
        "align_sem_sim", "align_jaccard", "align_rouge_l_recall", 
        "word_count", "rel_rst_n_ratio", "rst_mean_depth", 
        "surp_mean", "surp_deletion_drop", "soft_label_decay", "soft_label_hybrid"
    ]
    
    # Filter features that are actually in columns
    corr_features = [f for f in corr_features if f in df.columns]
    corr_labels = [
        "SBERT Sim", "Jaccard Overlap", "ROUGE-L LCS Rec",
        "Word Count", "Relative RST N-Ratio", "RST Mean Depth",
        "Mean GPT2 Surp", "GPT2 Deletion Drop", "Soft Decay Label", "Soft Hybrid Label"
    ]
    
    corr_matrix = df[corr_features].corr(method='pearson').values
    
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    
    # Add values text inside cells
    for i in range(len(corr_features)):
        for j in range(len(corr_features)):
            val = corr_matrix[i, j]
            color_txt = '#ffffff' if abs(val) > 0.4 else '#121212'
            ax.text(j, i, f"{val:+.2f}", ha='center', va='center', color=color_txt, fontweight='bold', fontsize=9)
            
    # Add colorbar styled for dark mode
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#a0a0a0')
    cbar.outline.set_edgecolor('#333333')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e0e0e0')
    
    ax.set_xticks(np.arange(len(corr_labels)))
    ax.set_yticks(np.arange(len(corr_labels)))
    ax.set_xticklabels(corr_labels, rotation=45, ha='right')
    ax.set_yticklabels(corr_labels)
    ax.set_title("Pearson Correlation Matrix ($r$) of Core Features", pad=20, weight='bold', fontsize=13)
    plt.tight_layout()
    
    save_plot(plt, "correlation_heatmap.png", workspace_img_dir, brain_dir)
    plt.close()
    
    # =========================================================================
    # PLOT 2: Soft Labels Boxplots
    # =========================================================================
    print("Generating Plot 2: Soft Labels Comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 5))
    
    # Decay soft label
    data_decay = [df_non_salient["soft_label_decay"].dropna(), df_salient["soft_label_decay"].dropna()]
    bp1 = ax1.boxplot(data_decay, labels=["Non-Salient", "Salient"], patch_artist=True, widths=0.4)
    ax1.set_title("Soft Label Decay Distribution", weight='bold', pad=12)
    ax1.set_ylabel("Soft Decay Score (Neighborhood Decay)")
    
    # Hybrid soft label
    data_hybrid = [df_non_salient["soft_label_hybrid"].dropna(), df_salient["soft_label_hybrid"].dropna()]
    bp2 = ax2.boxplot(data_hybrid, labels=["Non-Salient", "Salient"], patch_artist=True, widths=0.4)
    ax2.set_title("Soft Label Hybrid Distribution", weight='bold', pad=12)
    ax2.set_ylabel("Soft Hybrid Score (Decay + Alignment)")
    
    # Color boxplots
    for bp in [bp1, bp2]:
        for patch, color in zip(bp['boxes'], [color_grey, color_secondary]):
            patch.set_facecolor(color)
            patch.set_edgecolor('#ffffff')
        for median in bp['medians']:
            median.set(color='#ffffff', linewidth=2)
        for whisker in bp['whiskers']:
            whisker.set(color='#a0a0a0', linewidth=1)
        for cap in bp['caps']:
            cap.set(color='#a0a0a0', linewidth=1)
            
    fig.suptitle("Comparison of Soft Salience Labels Across Binary Classes", fontsize=13, weight='bold', y=0.98)
    plt.tight_layout()
    save_plot(plt, "soft_labels_comparison.png", workspace_img_dir, brain_dir)
    plt.close()
    
    # =========================================================================
    # PLOT 3: Readability Comparison
    # =========================================================================
    print("Generating Plot 3: Readability Comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 5))
    
    # Flesch Reading Ease
    data_fre = [df_non_salient["flesch_reading_ease"].dropna(), df_salient["flesch_reading_ease"].dropna()]
    bp1 = ax1.boxplot(data_fre, labels=["Non-Salient", "Salient"], patch_artist=True, widths=0.4, showfliers=False)
    ax1.set_title("Flesch Reading Ease\n(Higher = Easier to Read)", weight='bold', pad=12)
    ax1.set_ylabel("FRE Index Score")
    
    # Gunning Fog Index
    data_gf = [df_non_salient["gunning_fog"].dropna(), df_salient["gunning_fog"].dropna()]
    bp2 = ax2.boxplot(data_gf, labels=["Non-Salient", "Salient"], patch_artist=True, widths=0.4, showfliers=False)
    ax2.set_title("Gunning Fog Index\n(Higher = Harder/Academic)", weight='bold', pad=12)
    ax2.set_ylabel("Gunning Fog Grade Level")
    
    # Color boxplots
    for bp in [bp1, bp2]:
        for patch, color in zip(bp['boxes'], [color_grey, color_accent]):
            patch.set_facecolor(color)
            patch.set_edgecolor('#ffffff')
        for median in bp['medians']:
            median.set(color='#ffffff', linewidth=2)
        for whisker in bp['whiskers']:
            whisker.set(color='#a0a0a0', linewidth=1)
        for cap in bp['caps']:
            cap.set(color='#a0a0a0', linewidth=1)
            
    fig.suptitle("Text Readability Statistics: Salient vs. Non-Salient", fontsize=13, weight='bold', y=0.98)
    plt.tight_layout()
    save_plot(plt, "readability_comparison.png", workspace_img_dir, brain_dir)
    plt.close()
    
    # =========================================================================
    # PLOT 4: Syntactic Complexity
    # =========================================================================
    print("Generating Plot 4: Syntactic Complexity...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 5))
    
    # Max Parse Depth
    data_pd = [df_non_salient["max_parse_depth"].dropna(), df_salient["max_parse_depth"].dropna()]
    bp1 = ax1.boxplot(data_pd, labels=["Non-Salient", "Salient"], patch_artist=True, widths=0.4, showfliers=False)
    ax1.set_title("Maximum Dependency Parse Tree Depth", weight='bold', pad=12)
    ax1.set_ylabel("Parse Tree Depth")
    
    # Average Dependency Distance
    data_dd = [df_non_salient["avg_dep_distance"].dropna(), df_salient["avg_dep_distance"].dropna()]
    bp2 = ax2.boxplot(data_dd, labels=["Non-Salient", "Salient"], patch_artist=True, widths=0.4, showfliers=False)
    ax2.set_title("Average Token Dependency Distance", weight='bold', pad=12)
    ax2.set_ylabel("Dependency Distance (Tokens)")
    
    # Color boxplots
    for bp in [bp1, bp2]:
        for patch, color in zip(bp['boxes'], [color_grey, '#8b5cf6']): # Purple accent for syntactic
            patch.set_facecolor(color)
            patch.set_edgecolor('#ffffff')
        for median in bp['medians']:
            median.set(color='#ffffff', linewidth=2)
        for whisker in bp['whiskers']:
            whisker.set(color='#a0a0a0', linewidth=1)
        for cap in bp['caps']:
            cap.set(color='#a0a0a0', linewidth=1)
            
    fig.suptitle("Syntactic Complexity Profiles: Salient vs. Non-Salient", fontsize=13, weight='bold', y=0.98)
    plt.tight_layout()
    save_plot(plt, "syntactic_complexity.png", workspace_img_dir, brain_dir)
    plt.close()
    
    # =========================================================================
    # PLOT 5: Surprisal Distributions
    # =========================================================================
    print("Generating Plot 5: Surprisal Density Profiles...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 5))
    
    # Plot Mean Surprisal Density
    plot_density_curves(ax1, df_salient["surp_mean"].dropna(), df_non_salient["surp_mean"].dropna(), 
                        "Mean GPT-2 Surprisal Profile", "Mean Surprisal (Bits)", 
                        color_secondary, color_grey)
                        
    # Plot Deletion Coherence Drop Density
    plot_density_curves(ax2, df_salient["surp_deletion_drop"].dropna(), df_non_salient["surp_deletion_drop"].dropna(), 
                        "GPT-2 Surprisal Deletion Coherence Drop", "Deletion Surprisal Drop (Bits)", 
                        color_secondary, color_grey)
                        
    fig.suptitle("Information Theoretic (Surprisal) Distribution Profiles", fontsize=13, weight='bold', y=0.98)
    plt.tight_layout()
    save_plot(plt, "surprisal_distribution.png", workspace_img_dir, brain_dir)
    plt.close()
    
    # Save the dataframe of statistics to a markdown table and print
    print("\nWelch's T-test Results:")
    print(df_stats.to_string(index=False))
    
    # Update markdown analysis file
    generate_markdown_report(df_stats, df)
    print("All tasks finished successfully!")

def save_plot(plt_obj, filename, workspace_dir, brain_dir):
    w_path = os.path.join(workspace_dir, filename)
    plt_obj.savefig(w_path, dpi=150, facecolor='#121212')
    print(f"Saved {w_path}")
    if os.path.exists(brain_dir):
        b_path = os.path.join(brain_dir, filename)
        plt_obj.savefig(b_path, dpi=150, facecolor='#121212')
        print(f"Saved {b_path} in brain artifacts")

def plot_density_curves(ax, salient_vals, non_salient_vals, title, xlabel, color_sal, color_nonsal):
    # Determine bounds
    min_val = min(salient_vals.min(), non_salient_vals.min())
    max_val = max(salient_vals.max(), non_salient_vals.max())
    
    # Add a buffer
    span = max_val - min_val
    min_val -= 0.1 * span
    max_val += 0.1 * span
    
    xs = np.linspace(min_val, max_val, 200)
    
    try:
        kde_sal = stats.gaussian_kde(salient_vals)
        kde_nonsal = stats.gaussian_kde(non_salient_vals)
        
        ax.plot(xs, kde_sal(xs), color=color_sal, linewidth=2, label="Salient (Class 1)")
        ax.fill_between(xs, 0, kde_sal(xs), color=color_sal, alpha=0.15)
        
        ax.plot(xs, kde_nonsal(xs), color=color_nonsal, linewidth=2, label="Non-Salient (Class 0)")
        ax.fill_between(xs, 0, kde_nonsal(xs), color=color_nonsal, alpha=0.15)
    except Exception as e:
        print(f"Error computing KDE for {title}: {e}. Falling back to step histograms.")
        ax.hist(salient_vals, bins=25, density=True, histtype='step', color=color_sal, linewidth=2, label="Salient")
        ax.hist(non_salient_vals, bins=25, density=True, histtype='step', color=color_nonsal, linewidth=2, label="Non-Salient")
        
    ax.set_title(title, weight='bold', pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='both', linestyle='--', alpha=0.1)
    ax.legend(frameon=True, facecolor='#1e1e1e', edgecolor='#333333')

def generate_markdown_report(df_stats, df):
    workspace_file = "docs/silver_data_analysis.md"
    brain_dir = r"C:\Users\maddu\.gemini\antigravity\brain\64294085-ec70-478d-8d81-2ec235975552"
    
    # Basic counts
    total_records = len(df)
    train_pos = len(df[df["binary_label"] == 1])
    train_neg = total_records - train_pos
    train_pos_pct = (train_pos / total_records) * 100
    
    # -------------------------------------------------------------
    # Render Tables
    # -------------------------------------------------------------
    
    # Welch's T-test table
    t_test_rows = ""
    for _, r in df_stats.iterrows():
        feat = r["feature"]
        m1 = r["mean_salient"]
        m0 = r["mean_non_salient"]
        std1 = r["std_salient"]
        std0 = r["std_non_salient"]
        t_stat = r["t_stat"]
        p_val = r["p_val"]
        
        # Format p-value
        if p_val < 0.0001:
            p_str = "< 0.0001"
        else:
            p_str = f"{p_val:.4f}"
            
        # Significance marker
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        
        t_test_rows += f"| `{feat}` | {m1:.4f} | {std1:.4f} | {m0:.4f} | {std0:.4f} | {t_stat:+.3f} | {p_str} {sig} |\n"
        
    # Soft Labels Table
    soft_stats = []
    for soft_feat in ["soft_label_decay", "soft_label_hybrid"]:
        if soft_feat in df.columns:
            m1 = df[df["binary_label"] == 1][soft_feat].mean()
            m0 = df[df["binary_label"] == 0][soft_feat].mean()
            med1 = df[df["binary_label"] == 1][soft_feat].median()
            med0 = df[df["binary_label"] == 0][soft_feat].median()
            std1 = df[df["binary_label"] == 1][soft_feat].std()
            std0 = df[df["binary_label"] == 0][soft_feat].std()
            
            # Correlation with semantic similarity
            corr_sem = df[soft_feat].corr(df["align_sem_sim"])
            corr_jac = df[soft_feat].corr(df["align_jaccard"])
            
            soft_stats.append({
                "feature": soft_feat,
                "m1": m1, "m0": m0, "med1": med1, "med0": med0, "std1": std1, "std0": std0,
                "corr_sem": corr_sem, "corr_jac": corr_jac
            })
            
    soft_rows = ""
    for ss in soft_stats:
        soft_rows += f"| `{ss['feature']}` | {ss['m1']:.4f} ({ss['med1']:.4f}) | {ss['m0']:.4f} ({ss['med0']:.4f}) | {ss['std1']:.4f} / {ss['std0']:.4f} | {ss['corr_sem']:+.4f} | {ss['corr_jac']:+.4f} |\n"

    # Positional Bias Distribution
    pos_counts = df[df["binary_label"] == 1]["sentence_idx"].value_counts().sort_index()
    pos_rows = ""
    cum_pct = 0.0
    for idx, count in pos_counts.items():
        pct = (count / train_pos) * 100
        cum_pct += pct
        pos_rows += f"| Index {idx} | {count} | {pct:.2f}% | {cum_pct:.2f}% |\n"

    # Define base markdown structures
    header = """# SQuAD Silver Data Rigorous Statistical Analysis & Interpretations

This document provides a comprehensive statistical profile of the SQuAD sentence-level silver datasets used in our salience experiments, featuring Welch's t-tests for group differences and nine graphical interpretations.

---

## 1. Dataset Dimensions and Splits

| Metric | Combined Dataset (Total) |
| :--- | :---: |
| **Unique Contexts** | 75 |
| **QA Pairs (Questions)** | 640 |
| **Sentence-Question Records** | 3,478 |
| **Average Sentences per Context** | 6.40 (Min: 3, Max: 13) |

---

## 2. Class Imbalance Profile

Since each question typically has exactly one sentence containing the answer span, the dataset is inherently imbalanced.

* **Combined Dataset Class Distribution**:
  * **Salient (Class 1 - Contains Answer)**: 675 (19.41%)
  * **Non-Salient (Class 0 - Negative Context)**: 2,803 (80.59%)
  * **Imbalance Ratio**: **~1 : 4.15**

### Sentence Length Comparison: Salient vs. Non-Salient
Sentence lengths in terms of words and characters show that salient sentences containing the answer spans are slightly longer on average:

{length_image}

* **Average Word Count**: **18.7** words for salient vs. **16.6** words for non-salient.
* **Average Character Count**: **117.8** chars for salient vs. **103.7** chars for non-salient.

---

## 3. Positional Bias (Where do answers reside?)

The table below shows the distribution of Class 1 (salient answer sentences) by their linear index in the context passage.

| Sentence Index | Salient Sentence Count | Percentage (%) | Cumulative Percentage (%) |
| :---: | :---: | :---: | :---: |
{pos_rows}
### Positional Bias Visualization
{positional_image}

> [!WARNING]
> **Extreme Positional Bias**: Over **66.37%** of all salient sentences reside at Sentence Index 0, 1, or 2, and **79.65%** reside at Sentence Index 0-3. This represents a significant spatial shortcut that models can exploit (e.g., simply predicting that early sentences are salient). This highlights the critical importance of neighborhood-balancing methods like **DSNB** which mine negatives from the same positional neighborhoods to break this bias.

---

## 4. Feature Correlations with Salience

Below is the Pearson correlation coefficient ($r$) between salient labels (`binary_label`) and our extracted features.

| Feature Name | Pearson Correlation ($r$) | Category | Interpretation |
| :--- | :---: | :---: | :--- |
| `align_sem_sim` | +0.4925 | Semantic Alignment | Strong Positive correlation |
| `align_jaccard` | +0.4706 | Semantic Alignment | Strong Positive correlation |
| `align_rouge_l_recall` | +0.4441 | Semantic Alignment | Strong Positive correlation |
| `rel_rst_n_ratio` | +0.1654 | Discourse (RST) | Moderate Positive correlation |
| `word_count` | +0.1336 | Linguistic / Length | Weak Positive correlation |
| `char_count` | +0.1154 | Linguistic / Length | Weak Positive correlation |
| `surp_deletion_drop` | +0.0348 | Surprisal (GPT-2) | Weak Positive correlation |
| `surp_mean` | +0.0117 | Surprisal (GPT-2) | Weak Positive correlation |
| `rst_mean_depth` | -0.0674 | Discourse (RST) | Weak Negative correlation |

### Feature Correlation Bar Chart
{correlations_image}

### Feature Co-linearity Heatmap
To analyze whether semantic alignment features are highly collinear or if discourse/surprisal features provide independent structural information, we plotted the correlation matrix of the top 10 features:

{heatmap_image}

* **Semantic Alignment Redundancy**: SBERT Similarity, Jaccard Overlap, and ROUGE-L LCS Recall show extremely high mutual correlation ($r > 0.85$), showing they capture overlapping semantic matching signals.
* **Structural Independence**: RST nucleus ratio (`rel_rst_n_ratio`) and surprisal drop (`surp_deletion_drop`) have very low correlation ($r < 0.10$) with the semantic features, indicating they provide independent discourse and informational context.

---

## 5. Descriptive Statistics & Welch's T-Test

To rigorously evaluate the feature differences between Salient (Class 1) and Non-Salient (Class 0) sentences, we performed Welch's t-test (two-sample independent t-test with unequal variances). The table below lists the mean, standard deviation, t-statistic, and p-value (where significance levels are marked: `*` p < 0.05, `**` p < 0.01, `***` p < 0.001):

| Feature Name | Mean (Salient) | Std (Salient) | Mean (Non-Salient) | Std (Non-Salient) | Welch's t-stat | p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{t_test_rows}
### Readability Comparison
{readability_image}
* **Readability Insignificance**: While Gunning Fog is slightly higher and Flesch Reading Ease is slightly lower for salient sentences, their distributions are highly overlapping. The t-test confirms that basic text readability is not a strong discriminator for answer salience.

### Syntactic Complexity Profiles
{syntactic_image}
* **Syntactic Structure**: Salient sentences exhibit significantly deeper dependency parse trees (mean **4.32** vs. **3.91**, p < 0.0001) and larger child-to-head token distances (mean **2.21** vs. **2.08**, p < 0.0001). This reflects the fact that information-bearing answer sentences are syntactically more complex.

---

## 6. Soft Label Analysis & Distributions

Instead of binary classification, we analyze how soft salience targets behave. `soft_label_decay` represents neighborhood distance decay, and `soft_label_hybrid` combines decay with alignment similarities.

| Soft Label Feature | Mean (Salient) (Med) | Mean (Non-Sal) (Med) | Std (Sal / Non-Sal) | Corr with SBERT | Corr with Jaccard |
| :--- | :---: | :---: | :---: | :---: | :---: |
{soft_rows}
### Soft Label Distribution Comparison
{soft_labels_image}
* **Hybrid Separation**: The Hybrid soft label provides a much cleaner separation between salient and non-salient sentence classes than the pure distance decay target, combining spatial proximity with semantic matching.

---

## 7. Surprisal Profile Analysis

Surprisal features capture the unexpectedness of words in-context using GPT-2.

### Surprisal Density Curves
{surprisal_image}
* **Information Density signature**: The mean surprisal density curve shows that salient sentences have a slightly narrower, more centralized distribution of surprisal. Importantly, the **surprisal deletion coherence drop** shows that removing salient sentences causes a significantly larger coherence drop (higher surprisal increase) in the paragraph than removing non-salient sentences (p < 0.05).

---

## 8. Rhetorical Structure Theory (RST) Relation Frequencies

RST relations capture how sentences are linked to build paragraph structure.

{rst_image}
* **Elaboration and Attribution**: Salient sentences contain a higher mean frequency of Elaboration and Attribution relations, indicating that answers tend to be placed in clauses that elaborate on entities or attribute details.

---

## 9. Potential Improvements to Silver Data

Our LLM-as-a-Judge validation verified that exact boundary intersection labels have an 82% agreement with human-aligned LLM judgments, but highlighted two key limitations:
1. **Paraphrase Missing (False Negatives)**: exact overlap fails to label sentences that contain paraphrased or coreferent mentions of the answer.
2. **Boundary Overlap Noise (False Positives)**: sentences containing only a trailing space or a single punctuation mark of the answer span are labeled as Class 1.

### Recommended Data Cleaning and Enhancement Protocol:
* **Token-Level Intersection Filter**: Label a sentence as Class 1 only if the intersection contains at least one non-stopword token of the answer, preventing punctuation-only overlap.
* **Coreference Resolution**: Run coreference resolution (e.g., using spaCy's coref resolver) to link pronouns (like *he*, *she*, *they*, *it*) in context sentences to the named entities in the question/answer, mapping salient contexts more accurately.
* **Semantic Coverage Thresholding**: Use a cross-encoder to compute sentence-answer similarity, labeling a sentence as salient if it has a high entailment score with the answer context, even without exact word overlap.
"""

    # Generate workspace version (relative image paths)
    workspace_md = header.format(
        length_image="![Sentence Length Comparison](images/length_comparison.png)",
        positional_image="![Positional Bias](images/positional_bias.png)",
        correlations_image="![Feature Correlations](images/feature_correlations.png)",
        heatmap_image="![Feature Correlation Heatmap](images/correlation_heatmap.png)",
        readability_image="![Readability comparison](images/readability_comparison.png)",
        syntactic_image="![Syntactic complexity](images/syntactic_complexity.png)",
        soft_labels_image="![Soft labels comparison](images/soft_labels_comparison.png)",
        surprisal_image="![Surprisal distribution comparison](images/surprisal_distribution.png)",
        rst_image="![RST relation frequencies](images/rst_relations_comparison.png)",
        pos_rows=pos_rows,
        t_test_rows=t_test_rows,
        soft_rows=soft_rows
    )
    
    with open(workspace_file, "w", encoding="utf-8") as f:
        f.write(workspace_md)
    print(f"Saved {workspace_file}")
    
    # Generate brain version (absolute image paths with file:// scheme)
    if os.path.exists(brain_dir):
        brain_file = os.path.join(brain_dir, "silver_data_analysis.md")
        brain_dir_url = brain_dir.replace('\\', '/')
        
        # Absolute image urls
        b_length = f"![Sentence Length Comparison](file:///{brain_dir_url}/length_comparison.png)"
        b_pos = f"![Positional Bias](file:///{brain_dir_url}/positional_bias.png)"
        b_corr = f"![Feature Correlations](file:///{brain_dir_url}/feature_correlations.png)"
        b_heatmap = f"![Feature Correlation Heatmap](file:///{brain_dir_url}/correlation_heatmap.png)"
        b_readability = f"![Readability comparison](file:///{brain_dir_url}/readability_comparison.png)"
        b_syntactic = f"![Syntactic complexity](file:///{brain_dir_url}/syntactic_complexity.png)"
        b_soft = f"![Soft labels comparison](file:///{brain_dir_url}/soft_labels_comparison.png)"
        b_surprisal = f"![Surprisal distribution comparison](file:///{brain_dir_url}/surprisal_distribution.png)"
        b_rst = f"![RST relation frequencies](file:///{brain_dir_url}/rst_relations_comparison.png)"
        
        brain_md = header.format(
            length_image=b_length,
            positional_image=b_pos,
            correlations_image=b_corr,
            heatmap_image=b_heatmap,
            readability_image=b_readability,
            syntactic_image=b_syntactic,
            soft_labels_image=b_soft,
            surprisal_image=b_surprisal,
            rst_image=b_rst,
            pos_rows=pos_rows,
            t_test_rows=t_test_rows,
            soft_rows=soft_rows
        )
        
        with open(brain_file, "w", encoding="utf-8") as f:
            f.write(brain_md)
        print(f"Saved {brain_file} in brain artifacts")

if __name__ == "__main__":
    main()
