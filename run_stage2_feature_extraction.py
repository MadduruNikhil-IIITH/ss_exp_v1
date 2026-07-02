import os
import pickle
import time
import pandas as pd
from tqdm import tqdm
from src.linguistic_features import extract_linguistic_features
from src.surprisal_features import SurprisalCalculator
from src.rst_features import DiscourseParserWrapper, RSTFeatureExtractor
from src.concreteness_features import ConcretenessScoreCalculator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def run_stage2_pipeline(input_csv="squad_labeled_dataset.csv", cache_path="features_cache.pkl"):
    print("="*80)
    print("STAGE 2: FEATURE EXTRACTION PIPELINE (PSYCHFORMERS SURPRISAL & RST)")
    print("="*80)
    
    if not os.path.exists(input_csv):
        print(f"Error: Labeled dataset CSV '{input_csv}' not found. Run Stage 1 first!")
        return
        
    print(f"[{time.strftime('%X')}] Loading silver labeled records from '{input_csv}'...")
    df_all = pd.read_csv(input_csv)
    
    # 1. Initialize Feature Extractors
    print(f"[{time.strftime('%X')}] Initializing Stage 2 feature extractors (GPT-2 & BERT PLL & RST)...")
    start_init = time.time()
    device = "cuda"
    surprisal_calc = SurprisalCalculator(causal_model_name="gpt2", masked_model_name="bert-base-uncased", device=device)
    discourse_parser = DiscourseParserWrapper(device=device)
    rst_extractor = RSTFeatureExtractor()
    concrete_calc = ConcretenessScoreCalculator()
    sentiment_analyzer = SentimentIntensityAnalyzer()
    print(f"[{time.strftime('%X')}] Feature extractors initialized in {time.time() - start_init:.2f} seconds.\n")
    
    def process_split(df_split, split_name):
        if df_split.empty:
            print(f"No records for {split_name} split.")
            return []
            
        unique_contexts = df_split["context"].unique()
        print(f"[{time.strftime('%X')}] Extracting features for {len(unique_contexts)} unique contexts in {split_name}...")
        
        context_features_cache = {}
        start_extract = time.time()
        
        for ctx_idx, ctx in enumerate(tqdm(unique_contexts, desc=f"Stage 2 ({split_name}) contexts")):
            ctx_rows = df_split[df_split["context"] == ctx]
            
            # Reconstruct sentence boundary objects
            sentences = []
            seen_indices = set()
            for _, row in ctx_rows.iterrows():
                idx = int(row["sentence_idx"])
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    sentences.append({
                        "sentence_idx": idx,
                        "text": row["sentence_text"],
                        "start_char": int(row["start_char"]),
                        "end_char": int(row["end_char"])
                    })
            sentences = sorted(sentences, key=lambda x: x["sentence_idx"])
            
            # A. Extract Causal Surprisal and Masked PLL (GPT-2 & BERT PLL)
            try:
                surp_feats = surprisal_calc.extract_surprisal_features(sentences, ctx)
            except Exception as e:
                print(f"\n[Warning] Surprisal error on context {ctx_idx}: {e}")
                surp_feats = [{} for _ in sentences]
                
            # B. Extract RST Discourse features
            try:
                rst_root = discourse_parser.parse(ctx, sentences)
                rst_feats = rst_extractor.extract_rst_features(rst_root, sentences)
            except Exception as e:
                print(f"\n[Warning] RST error on context {ctx_idx}: {e}")
                rst_feats = [{} for _ in sentences]
                
            context_features_cache[ctx] = (surp_feats, rst_feats)
            
        print(f"[{time.strftime('%X')}] Split context features extracted in {time.time() - start_extract:.2f} seconds.")
        
        # Now, process sentence-level and merge all features
        print(f"Mapping and merging features to sentence records...")
        final_records = []
        grouped_by_q = df_split.groupby("question_id")
        
        for q_id, group in tqdm(grouped_by_q, desc=f"Mapping QA records ({split_name})"):
            first_row = group.iloc[0]
            question = first_row["question"]
            ctx = first_row["context"]
            
            surp_feats, rst_feats = context_features_cache.get(ctx, (None, None))
            
            # Build sentences list
            sentences = []
            for _, row in group.iterrows():
                sentences.append({
                    "sentence_idx": int(row["sentence_idx"]),
                    "text": row["sentence_text"],
                    "start_char": int(row["start_char"]),
                    "end_char": int(row["end_char"])
                })
            sentences = sorted(sentences, key=lambda x: x["sentence_idx"])
            
            # C. Extract Linguistic features per sentence
            ling_feats = []
            for sent in sentences:
                try:
                    lf = extract_linguistic_features(sent["text"])
                    # Extract concreteness features
                    cf = concrete_calc.extract_concreteness_features(sent["text"])
                    lf.update(cf)
                    # Extract sentiment polarity features
                    s_scores = sentiment_analyzer.polarity_scores(sent["text"])
                    lf["sentiment_polarity_pos"] = float(s_scores["pos"])
                    lf["sentiment_polarity_neg"] = float(s_scores["neg"])
                    lf["sentiment_polarity_neu"] = float(s_scores["neu"])
                    lf["sentiment_polarity_compound"] = float(s_scores["compound"])
                except Exception as e:
                    lf = {}
                ling_feats.append(lf)
                
            # Merge
            for idx, (_, row) in enumerate(group.iterrows()):
                sent_idx = int(row["sentence_idx"])
                
                s_ling = ling_feats[idx] if idx < len(ling_feats) else {}
                s_surp = surp_feats[sent_idx] if surp_feats and sent_idx < len(surp_feats) else {}
                s_rst = rst_feats[sent_idx] if rst_feats and sent_idx < len(rst_feats) else {}
                
                combined_features = {}
                combined_features.update(s_ling)
                combined_features.update(s_surp)
                combined_features.update(s_rst)
                
                final_records.append({
                    "question_id": q_id,
                    "question": question,
                    "context": ctx,
                    "sentence_idx": sent_idx,
                    "sentence_text": row["sentence_text"],
                    "binary_label": int(row["binary_label"]),
                    "soft_label_decay": float(row["soft_label_decay"]),
                    "soft_label_hybrid": float(row["soft_label_hybrid"]),
                    "features": combined_features
                })
                
        return final_records

    # Process train and validation splits separately
    train_df = df_all[df_all["split"] == "train"]
    val_df = df_all[df_all["split"] == "validation"]
    
    train_records = process_split(train_df, "train")
    val_records = process_split(val_df, "validation")
    
    # Save cache
    print(f"\n[{time.strftime('%X')}] Saving completed features cache to '{cache_path}'...")
    cache_data = {
        "train": train_records,
        "validation": val_records
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
        
    print(f"[{time.strftime('%X')}] Stage 2 complete! Cached {len(train_records)} train and {len(val_records)} val records.")
    
    # Automatically generate statistical plots
    generate_stage2_plots(cache_path=cache_path)
    
    # Automatically generate dataset balancing sizes CSV
    generate_balancing_sizes_csv(train_records)
    print("="*80)

def generate_stage2_plots(cache_path="features_cache.pkl"):
    """
    Generates all diagnostic statistical plots and saves them to docs/images/
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Warning] matplotlib or numpy not found. Skipping plot generation.")
        return

    print("Automatically generating dataset statistics plots...")
    if not os.path.exists(cache_path):
        print(f"Error: Cache file '{cache_path}' not found.")
        return

    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    all_records = cache_data["train"] + cache_data["validation"]

    df_list = []
    for r in all_records:
        row = {
            "binary_label": r["binary_label"],
            "sentence_text": r["sentence_text"],
            "question_id": r["question_id"],
            "sentence_idx": r["sentence_idx"],
            "soft_label_hybrid": r["soft_label_hybrid"],
            "soft_label_decay": r["soft_label_decay"],
        }
        for k, v in r["features"].items():
            row[k] = v
        df_list.append(row)
    df = pd.DataFrame(df_list)

    # Create images directory
    os.makedirs(os.path.join("docs", "images"), exist_ok=True)

    def save_plot(name):
        plt.tight_layout()
        plt.savefig(os.path.join("docs", "images", name), dpi=150)
        plt.close()
        print(f"  Saved plot: {name}")

    # 1. Positional Bias (positional_bias.png)
    plt.figure(figsize=(7, 4.5))
    salient_df = df[df["binary_label"] == 1]
    counts = salient_df["sentence_idx"].value_counts().sort_index()
    percentages = (counts / len(salient_df)) * 100
    
    indices = list(range(8))
    values = [percentages.get(idx, 0.0) for idx in indices]

    bars = plt.bar(indices, values, color="#3498db", edgecolor="black", alpha=0.8)
    plt.title("Distribution of Salient Sentences by Linear Position", fontsize=12, fontweight="bold")
    plt.xlabel("Sentence Index in Context Paragraph", fontsize=10)
    plt.ylabel("Percentage of Answers (%)", fontsize=10)
    plt.xticks(indices)
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.1f}%",
                    (bar.get_x() + bar.get_width() / 2., height),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                    textcoords="offset points")
    plt.ylim(0, 35)
    save_plot("positional_bias.png")

    # Helper for boxplots
    def make_boxplot(y_col, title, ylabel, name):
        plt.figure(figsize=(6, 4.5))
        salient_data = df[df["binary_label"] == 1][y_col].dropna()
        non_salient_data = df[df["binary_label"] == 0][y_col].dropna()
        
        # Check if tick_labels parameter should be used (Matplotlib >= 3.9)
        try:
            box = plt.boxplot([salient_data, non_salient_data], tick_labels=["Salient", "Non-Salient"], 
                                patch_artist=True, widths=0.4)
        except TypeError:
            box = plt.boxplot([salient_data, non_salient_data], labels=["Salient", "Non-Salient"], 
                                patch_artist=True, widths=0.4)
        
        colors = ["#f1c40f", "#3498db"]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        plt.title(title, fontsize=12, fontweight="bold")
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6, axis="y")
        save_plot(name)

    # 2. Length Comparison (length_comparison.png)
    if "word_count" in df.columns:
        make_boxplot("word_count", "Sentence Length: Salient vs. Non-Salient", "Word Count", "length_comparison.png")

    # 3. Readability Comparison (readability_comparison.png)
    if "gunning_fog" in df.columns:
        make_boxplot("gunning_fog", "Text Readability (Gunning Fog Index)", "Gunning Fog Score", "readability_comparison.png")

    # 4. Syntactic Complexity (syntactic_complexity.png)
    if "avg_dep_distance" in df.columns:
        make_boxplot("avg_dep_distance", "Syntactic Complexity: Dependency Distance", "Average Dependency Parse Distance", "syntactic_complexity.png")

    # 5. Surprisal Distribution (surprisal_distribution.png)
    if "surp_deletion_drop" in df.columns:
        plt.figure(figsize=(7, 4.5))
        salient_data = df[df["binary_label"] == 1]["surp_deletion_drop"].dropna()
        non_salient_data = df[df["binary_label"] == 0]["surp_deletion_drop"].dropna()
        
        plt.hist(salient_data, bins=20, density=True, alpha=0.4, color="#f1c40f", label="Salient", histtype="stepfilled")
        plt.hist(non_salient_data, bins=20, density=True, alpha=0.4, color="#3498db", label="Non-Salient", histtype="stepfilled")
        plt.title("Information Density: Surprisal Deletion Drop", fontsize=12, fontweight="bold")
        plt.xlabel("Surprisal Coherence Drop (Bits)", fontsize=10)
        plt.ylabel("Probability Density", fontsize=10)
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        save_plot("surprisal_distribution.png")

    # 6. Soft Label Target Separability (soft_labels_comparison.png)
    make_boxplot("soft_label_hybrid", "Soft Hybrid Target Label Separability", "Soft Target Value", "soft_labels_comparison.png")

    # 7. Correlation Heatmap (correlation_heatmap.png)
    top_feats = [
        "binary_label", "align_sem_sim", "align_jaccard",
        "rst_n_count", "rst_s_count", "rst_mean_depth",
        "surp_deletion_drop", "rel_surp_causal_pf_sum_ratio",
        "word_count", "avg_dep_distance"
    ]
    top_feats = [f for f in top_feats if f in df.columns]
    df_corr = df[top_feats].corr()
    
    names_map = {
        "binary_label": "Salience Target",
        "align_sem_sim": "SBERT Alignment",
        "align_jaccard": "Jaccard Overlap",
        "rst_n_count": "RST Nuclei",
        "rst_s_count": "RST Satellites",
        "rst_mean_depth": "RST Parse Depth",
        "surp_deletion_drop": "Surprisal Drop",
        "rel_surp_causal_pf_sum_ratio": "Info Density",
        "word_count": "Word Count",
        "avg_dep_distance": "Dep Distance"
    }
    df_corr = df_corr.rename(columns=names_map, index=names_map)

    plt.figure(figsize=(9, 8))
    cax = plt.imshow(df_corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(cax)
    
    ticks = np.arange(len(df_corr.columns))
    plt.xticks(ticks, df_corr.columns, rotation=45, ha="right", fontsize=9)
    plt.yticks(ticks, df_corr.index, fontsize=9)
    
    for i in range(len(df_corr.index)):
        for j in range(len(df_corr.columns)):
            plt.text(j, i, f"{df_corr.values[i, j]:.2f}",
                     ha="center", va="center", color="white" if abs(df_corr.values[i, j]) > 0.4 else "black", fontsize=9)
            
    plt.title("Pearson Correlation Matrix of Key Features", fontsize=12, fontweight="bold")
    save_plot("correlation_heatmap.png")

def generate_balancing_sizes_csv(train_records):
    """
    Calculates dataset sizes for all balancing methods and saves them to docs/dataset_balancing_sizes.csv
    """
    print("Generating dataset balancing sizes CSV...")
    try:
        from src.data_processing import (
            apply_pairwise_balancing,
            apply_cluster_balancing,
            apply_rst_balancing,
            apply_dsnb_balancing
        )
    except ImportError:
        print("[Warning] Could not import dataset balancing functions. Skipping CSV generation.")
        return

    # Calculate sizes
    none_len = len(train_records)
    none_salient = sum(1 for r in train_records if r["binary_label"] == 1)
    none_non_salient = none_len - none_salient
    
    # Extract feature keys
    all_features = list(train_records[0]["features"].keys()) if train_records else []
    
    # Balancing methods
    pw_len = len(apply_pairwise_balancing(train_records))
    cl_len = len(apply_cluster_balancing(train_records, all_features))
    rst_len = len(apply_rst_balancing(train_records))
    dsnb_len = len(apply_dsnb_balancing(train_records))
    
    # Class allocations
    data = [
        {
            "Balancing Method": "None (Unbalanced Raw)",
            "Training Samples": none_len,
            "Salient (Class 1)": none_salient,
            "Non-Salient (Class 0)": none_non_salient,
            "Label Balance": f"{(none_salient/none_len)*100:.2f}% / {(none_non_salient/none_len)*100:.2f}%"
        },
        {
            "Balancing Method": "Pairwise",
            "Training Samples": pw_len,
            "Salient (Class 1)": pw_len // 2 + (pw_len % 2),
            "Non-Salient (Class 0)": pw_len // 2,
            "Label Balance": "50.00% / 50.00%"
        },
        {
            "Balancing Method": "Cluster",
            "Training Samples": cl_len,
            "Salient (Class 1)": cl_len // 2,
            "Non-Salient (Class 0)": cl_len // 2,
            "Label Balance": "50.00% / 50.00%"
        },
        {
            "Balancing Method": "RST-Neighborhood",
            "Training Samples": rst_len,
            "Salient (Class 1)": rst_len // 2,
            "Non-Salient (Class 0)": rst_len // 2,
            "Label Balance": "50.00% / 50.00%"
        },
        {
            "Balancing Method": "DSNB (Proposed)",
            "Training Samples": dsnb_len,
            "Salient (Class 1)": dsnb_len // 2,
            "Non-Salient (Class 0)": dsnb_len // 2,
            "Label Balance": "50.00% / 50.00%"
        }
    ]
    
    df_sizes = pd.DataFrame(data)
    csv_path = os.path.join("docs", "dataset_balancing_sizes.csv")
    df_sizes.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

if __name__ == "__main__":
    run_stage2_pipeline()
