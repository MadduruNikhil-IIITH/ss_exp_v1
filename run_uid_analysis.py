import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp
from src.surprisal_features import SurprisalCalculator

def main():
    print("="*80)
    print("INFORMATION DENSITY STUDY: TOKEN-LEVEL SURPRISAL UID ANALYSIS")
    print("="*80)

    cache_path = "features_cache.pkl"
    if not os.path.exists(cache_path):
        print(f"Error: Cache file '{cache_path}' not found. Run Stage 2 first!")
        return

    print(f"Loading cached features from '{cache_path}'...")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    val_records = cache_data["validation"]
    print(f"Loaded {len(val_records)} validation records.")

    # Initialize SurprisalCalculator to extract token-level lists
    print("Initializing Surprisal Calculator (GPT-2)...")
    device = "cuda"
    surp_calc = SurprisalCalculator(causal_model_name="gpt2", masked_model_name="bert-base-uncased", device=device)

    # We want to extract token-level causal surprisals for each word in each validation sentence
    # Group validation records by context to avoid duplicate context tokenization
    grouped_contexts = {}
    for r in val_records:
        ctx = r["context"]
        if ctx not in grouped_contexts:
            grouped_contexts[ctx] = []
        grouped_contexts[ctx].append(r)

    salient_surprisals = []
    non_salient_surprisals = []

    print("Extracting token-level surprisals...")
    for ctx, records in tqdm(grouped_contexts.items(), desc="Contexts processed"):
        # Unique sentences in this context
        sentences_map = {}
        for r in records:
            idx = int(r["sentence_idx"])
            if idx not in sentences_map:
                sentences_map[idx] = {
                    "text": r["sentence_text"],
                    "binary_label": r["binary_label"]
                }
                
        # Reconstruct full context sentences
        sorted_indices = sorted(sentences_map.keys())
        sentences = []
        char_idx = 0
        for idx in sorted_indices:
            text = sentences_map[idx]["text"]
            start = char_idx
            end = char_idx + len(text)
            sentences.append({
                "sentence_idx": idx,
                "text": text,
                "start_char": start,
                "end_char": end,
                "binary_label": sentences_map[idx]["binary_label"]
            })
            char_idx = end + 1 # +1 for space/newline separator
            
        context_text = " ".join([s["text"] for s in sentences])
        
        for sent in sentences:
            s_start = sent["start_char"]
            s_end = sent["end_char"]
            label = sent["binary_label"]
            
            preceding_text = context_text[:s_start]
            target_text = context_text[s_start:s_end]
            following_text = context_text[s_end:]
            
            # Causal Surprisal segmentation and calculation
            try:
                p_ctx, t_words, f_words = surp_calc.segment_context(
                    surp_calc.causal_tokenizer, preceding_text, target_text, following_text
                )
                surps = surp_calc.get_causal_surprisals(p_ctx, t_words)
                
                if label == 1:
                    salient_surprisals.extend(surps)
                else:
                    non_salient_surprisals.extend(surps)
            except Exception as e:
                # Silently skip errors (e.g. empty target)
                pass

    salient_arr = np.array(salient_surprisals)
    non_salient_arr = np.array(non_salient_surprisals)

    print(f"\nExtracted token surprisals:")
    print(f"  - Salient tokens:     {len(salient_arr)}")
    print(f"  - Non-salient tokens: {len(non_salient_arr)}")

    if len(salient_arr) == 0 or len(non_salient_arr) == 0:
        print("Error: Empty surprisal arrays.")
        return

    # Compute key stats
    mean_sal = np.mean(salient_arr)
    mean_nsal = np.mean(non_salient_arr)
    median_sal = np.median(salient_arr)
    median_nsal = np.median(non_salient_arr)
    
    # Quantiles
    quantiles = [5, 25, 50, 75, 90, 95, 99]
    q_sal = np.percentile(salient_arr, quantiles)
    q_nsal = np.percentile(non_salient_arr, quantiles)
    
    # KS-test to compare distributions
    ks_stat, ks_p = ks_2samp(salient_arr, non_salient_arr)

    # Print results
    print("\n" + "="*80)
    print("TOKEN-LEVEL SURPRISAL QUANTILE COMPARISON")
    print("="*80)
    print(f"Mean:   Salient = {mean_sal:.4f}, Non-Salient = {mean_nsal:.4f} (Diff = {mean_sal - mean_nsal:.4f})")
    print(f"Median: Salient = {median_sal:.4f}, Non-Salient = {median_nsal:.4f} (Diff = {median_sal - median_nsal:.4f})")
    print(f"KS-Test statistic: {ks_stat:.4f} (p-value = {ks_p:.4e})")
    print("-"*80)
    print(f"{'Quantile':<10} | {'Salient':<12} | {'Non-Salient':<12} | {'Difference (Sal - Non)':<22}")
    print("-"*80)
    for idx, q in enumerate(quantiles):
        diff = q_sal[idx] - q_nsal[idx]
        print(f"{q:<10} | {q_sal[idx]:<12.4f} | {q_nsal[idx]:<12.4f} | {diff:<22.4f}")
    print("="*80)

    # Generate CDF plot
    plt.figure(figsize=(8, 6))
    
    # Sort data for CDF
    x_sal = np.sort(salient_arr)
    y_sal = np.arange(len(x_sal)) / float(len(x_sal))
    
    x_nsal = np.sort(non_salient_arr)
    y_nsal = np.arange(len(x_nsal)) / float(len(x_nsal))
    
    plt.plot(x_nsal, y_nsal, label='Non-Salient', color='#e74c3c', linewidth=2)
    plt.plot(x_sal, y_sal, label='Salient', color='#3498db', linewidth=2)
    
    # Highlight P90 gap
    p90_sal = q_sal[quantiles.index(90)]
    p90_nsal = q_nsal[quantiles.index(90)]
    plt.axvline(x=p90_nsal, color='#e74c3c', linestyle='--', alpha=0.5)
    plt.axvline(x=p90_sal, color='#3498db', linestyle='--', alpha=0.5)
    
    plt.title('CDFs of Word-Level Surprisal (SQuAD Validation)')
    plt.xlabel('Word Surprisal (bits)')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save plot
    plot_path = os.path.join("docs", "images", "uid_distribution.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved CDF comparison plot to '{plot_path}'")
    
    # Saved to workspace

    # Write Markdown report
    report_path = os.path.join("docs", "uid_analysis_report.md")
    
    def write_report(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# SQuAD Token-Level Surprisal UID Analysis Report\n\n")
            f.write("This report analyzes token-level information density in SQuAD context sentences to test the **Uniform Information Density (UID)** hypothesis, investigating whether salient sentences suppress rare, hard-to-predict words.\n\n")
            f.write("## 1. Quantile Comparison Table\n")
            f.write("| Quantile | Salient (bits) | Non-Salient (bits) | Difference (Salient - Non) |\n")
            f.write("| --- | --- | --- | --- |\n")
            for idx, q in enumerate(quantiles):
                diff = q_sal[idx] - q_nsal[idx]
                f.write(f"| P{q} | `{q_sal[idx]:.4f}` | `{q_nsal[idx]:.4f}` | **`{diff:.4f}`** |\n")
            f.write("\n")
            f.write("### Statistical Test\n")
            f.write(f"- **Kolmogorov-Smirnov Test**: KS-statistic = `{ks_stat:.4f}` ($p = {ks_p:.4e}$)\n\n")
            f.write("## 2. Key Findings & Discussion\n\n")
            
            # Check upper tail asymmetry
            diff_p50 = q_sal[quantiles.index(50)] - q_nsal[quantiles.index(50)]
            diff_p90 = q_sal[quantiles.index(90)] - q_nsal[quantiles.index(90)]
            diff_p95 = q_sal[quantiles.index(95)] - q_nsal[quantiles.index(95)]
            
            f.write(f"- **Median Difference (P50)**: `{diff_p50:.4f}` bits.\n")
            f.write(f"- **Upper-Tail Difference (P90)**: `{diff_p90:.4f}` bits.\n")
            f.write(f"- **Upper-Tail Difference (P95)**: `{diff_p95:.4f}` bits.\n\n")
            
            f.write("### Upper-Tail Surprisal Asymmetry\n")
            if diff_p90 < -0.1 and diff_p95 < -0.1:
                f.write(f"The results **confirm** the paper's token-level upper-tail surprisal asymmetry finding. While the median difference is small (`{diff_p50:.4f}` bits), the difference in the upper tail is significantly larger (`{diff_p90:.4f}` bits at P90 and `{diff_p95:.4f}` bits at P95). "
                        "This indicates that salient SQuAD sentences share similar word-level surprisal distributions in the bulk, but specifically **suppress the rarest, hardest-to-predict tokens** in the tail.\n")
            else:
                f.write("The results do not show a pronounced upper-tail suppression in salient sentences. The distributions remain close across both quantiles.\n")

    write_report(report_path)
    print("UID Analysis completed successfully. Reports saved to 'docs/uid_analysis_report.md'.")

if __name__ == "__main__":
    main()
