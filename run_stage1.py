import os
import pickle
import pandas as pd
from tqdm import tqdm
from src.data_processing import build_silver_squad_dataset
from src.linguistic_features import extract_linguistic_features
from src.surprisal_features import SurprisalCalculator
from src.rst_features import DiscourseParserWrapper, RSTFeatureExtractor
from src.alignment_features import AlignmentFeatureExtractor

def run_stage1_pipeline(num_train_contexts=100, num_val_contexts=30, cache_path="features_cache.pkl"):
    """
    Orchestrates the Stage 1 pipeline: SQuAD loading, feature extraction, and caching.
    """
    print("="*60)
    print("STAGE 1: FEATURE EXTRACTION PIPELINE START")
    print("="*60)
    
    # 1. Initialize Feature Extractors
    print("Initializing feature extractors...")
    device = "cuda"
    surprisal_calc = SurprisalCalculator(model_name="gpt2", device=device)
    discourse_parser = DiscourseParserWrapper(device=device)
    rst_extractor = RSTFeatureExtractor()
    alignment_extractor = AlignmentFeatureExtractor(sbert_model_name="all-MiniLM-L6-v2", device=device)
    
    def process_split(num_contexts, split_name):
        print(f"\n--- Processing {split_name} split ---")
        # Build raw silver dataset (sentences, offsets, and labels)
        df = build_silver_squad_dataset(num_contexts=num_contexts, split=split_name)
        if df.empty:
            print(f"No records extracted for {split_name}.")
            return []
            
        # Group by context to de-duplicate heavy processing (RST and GPT-2)
        unique_contexts = df["context"].unique()
        print(f"Extracting features for {len(unique_contexts)} unique context passages...")
        
        # We will build a mapping of context -> (surprisal_features_list, rst_features_list)
        context_features_cache = {}
        for ctx in tqdm(unique_contexts, desc=f"Extracting context features ({split_name})"):
            # Get sentence boundaries for this context
            ctx_rows = df[df["context"] == ctx]
            first_row = ctx_rows.iloc[0]
            
            # Reconstruct sentences boundaries for this context
            sentences = []
            seen_indices = set()
            for _, row in ctx_rows.iterrows():
                idx = row["sentence_idx"]
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    sentences.append({
                        "sentence_idx": idx,
                        "text": row["sentence_text"],
                        "start_char": row["start_char"],
                        "end_char": row["end_char"]
                    })
            sentences = sorted(sentences, key=lambda x: x["sentence_idx"])
            
            # A. Surprisal features (GPT-2 context run)
            try:
                tokens = surprisal_calc.get_token_surprisals(ctx)
                surp_feats = surprisal_calc.extract_surprisal_features(tokens, sentences)
            except Exception as e:
                print(f"\nError extracting surprisal: {e}")
                surp_feats = [{} for _ in sentences]
                
            # B. RST features (isanlp_rst context run)
            try:
                rst_root = discourse_parser.parse(ctx, sentences)
                rst_feats = rst_extractor.extract_rst_features(rst_root, sentences)
            except Exception as e:
                print(f"\nError extracting RST: {e}")
                rst_feats = [{} for _ in sentences]
                
            context_features_cache[ctx] = (surp_feats, rst_feats)
            
        # Now, process each row (question-sentence pair)
        print("Extracting sentence-level and alignment features...")
        final_records = []
        
        # Group by question_id to optimize alignment encoding (only run SBERT once per question)
        grouped_by_q = df.groupby("question_id")
        
        for q_id, group in tqdm(grouped_by_q, desc=f"Processing QA pairs ({split_name})"):
            first_row = group.iloc[0]
            question = first_row["question"]
            ctx = first_row["context"]
            
            # Retrieve cached context-level features
            surp_feats, rst_feats = context_features_cache.get(ctx, (None, None))
            
            # Generate sentence boundaries list
            sentences = []
            for _, row in group.iterrows():
                sentences.append({
                    "sentence_idx": row["sentence_idx"],
                    "text": row["sentence_text"],
                    "start_char": row["start_char"],
                    "end_char": row["end_char"]
                })
            sentences = sorted(sentences, key=lambda x: x["sentence_idx"])
            
            # C. Question-Sentence Alignment features
            try:
                align_feats = alignment_extractor.extract_alignment_features(question, sentences)
            except Exception as e:
                print(f"\nError extracting alignment: {e}")
                align_feats = [{} for _ in sentences]
                
            # D. Linguistic features (run per sentence)
            ling_feats = []
            for sent in sentences:
                try:
                    lf = extract_linguistic_features(sent["text"])
                except Exception as e:
                    print(f"\nError extracting linguistic features: {e}")
                    lf = {}
                ling_feats.append(lf)
                
            # Combine all features for each sentence in the group
            for idx, (_, row) in enumerate(group.iterrows()):
                sent_idx = row["sentence_idx"]
                
                # Retrieve features corresponding to this sentence index
                s_ling = ling_feats[idx] if idx < len(ling_feats) else {}
                s_align = align_feats[idx] if idx < len(align_feats) else {}
                s_surp = surp_feats[sent_idx] if surp_feats and sent_idx < len(surp_feats) else {}
                s_rst = rst_feats[sent_idx] if rst_feats and sent_idx < len(rst_feats) else {}
                
                # Merge all features into a flat dictionary
                combined_features = {}
                combined_features.update(s_ling)
                combined_features.update(s_surp)
                combined_features.update(s_rst)
                combined_features.update(s_align)
                
                final_records.append({
                    "question_id": q_id,
                    "question": question,
                    "context": ctx,
                    "sentence_idx": sent_idx,
                    "sentence_text": row["sentence_text"],
                    "binary_label": row["binary_label"],
                    "soft_label_decay": row["soft_label_decay"],
                    "soft_label_hybrid": row["soft_label_hybrid"],
                    "features": combined_features
                })
                
        return final_records

    # Process Train and Val splits
    train_records = process_split(num_train_contexts, "train")
    val_records = process_split(num_val_contexts, "validation")
    
    # Save to Cache
    print(f"\nSaving extracted features to '{cache_path}'...")
    cache_data = {
        "train": train_records,
        "validation": val_records
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
        
    print(f"Stage 1 complete! Cached {len(train_records)} training records and {len(val_records)} validation records.")
    print("="*60)

if __name__ == "__main__":
    # Run pipeline on a small sample first to verify correctness
    # Using 10 training contexts and 3 validation contexts for verification
    run_stage1_pipeline(num_train_contexts=10, num_val_contexts=3, cache_path="features_cache.pkl")
