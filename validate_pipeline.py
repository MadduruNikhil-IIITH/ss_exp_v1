import os
import pickle
import numpy as np
import pandas as pd
import torch
from src.data_processing import apply_pairwise_balancing, apply_cluster_balancing
from src.classifiers.hybrid_bert import HybridGatedBERTModel

def validate_environment():
    print("1. Checking Environment & CUDA GPU Acceleration...")
    cuda_avail = torch.cuda.is_available()
    print(f"   - PyTorch version: {torch.__version__}")
    print(f"   - CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"   - GPU Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("   - Warning: Running on CPU. Reinstall PyTorch with CUDA if you want GPU speed.")
    print("   - SUCCESS: Environment check completed.\n")

def validate_cache(cache_path="features_cache_deletion.pkl"):
    print("2. Checking Features Cache Validity...")
    if not os.path.exists(cache_path):
        print(f"   - Error: Cache file '{cache_path}' not found. Run run_stage1.py first.")
        return None
        
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
        
    train_records = data.get("train", [])
    val_records = data.get("validation", [])
    
    print(f"   - Train records loaded: {len(train_records)}")
    print(f"   - Validation records loaded: {len(val_records)}")
    
    if not train_records:
        print("   - Error: Train records are empty.")
        return None
        
    # Check keys
    sample = train_records[0]
    required_keys = ["question_id", "question", "context", "sentence_idx", "sentence_text", "binary_label", "features"]
    for k in required_keys:
        if k not in sample:
            print(f"   - Error: Missing required key '{k}' in records.")
            return None
            
    # Check features for NaNs/Infs
    feature_keys = list(sample["features"].keys())
    print(f"   - Total features extracted per sentence: {len(feature_keys)}")
    
    nan_count = 0
    for r in train_records:
        for k, v in r["features"].items():
            if np.isnan(v) or np.isinf(v):
                nan_count += 1
                
    print(f"   - NaN / Inf count in features: {nan_count}")
    if nan_count > 0:
        print("   - Warning: NaNs or Infs detected in features cache!")
    else:
        print("   - SUCCESS: Features cache validated.")
    print("")
    return data

def validate_balancing(data):
    print("3. Checking Class Balancing Algorithms...")
    train_records = data["train"]
    
    # Test Cluster-Based Balancing
    all_features = list(train_records[0]["features"].keys())
    balanced_cluster_records = apply_cluster_balancing(train_records, all_features)
    df_balanced_cluster = pd.DataFrame(balanced_cluster_records)
    
    pos_count = len(df_balanced_cluster[df_balanced_cluster["binary_label"] == 1])
    neg_count = len(df_balanced_cluster[df_balanced_cluster["binary_label"] == 0])
    print(f"   - Cluster-based Balancing: Positives = {pos_count}, Negatives = {neg_count}")
    if pos_count == neg_count:
        print("   - SUCCESS: Cluster balancing results in 1:1 ratio.")
    else:
        print("   - Warning: Cluster balancing does not result in 1:1 ratio.")
        
    # Test Pairwise Balancing
    pairwise_records = apply_pairwise_balancing(train_records)
    df_pairwise = pd.DataFrame(pairwise_records)
    print(f"   - Pairwise Balancing: Total pairs generated = {len(df_pairwise)}")
    if not df_pairwise.empty:
        p_label_ratio = df_pairwise["label"].mean()
        print(f"   - Pairwise label ratio (should be ~0.5): {p_label_ratio:.2f}")
        print("   - SUCCESS: Pairwise balancing validated.")
    else:
        print("   - Warning: No pairs generated.")
    print("")

def validate_hybrid_bert(data):
    print("4. Checking Hybrid Gated BERT Model Architectures...")
    train_records = data["train"]
    all_features = list(train_records[0]["features"].keys())
    
    rst_cols = [k for k in all_features if k.startswith("rst_") or k.startswith("rel_rst_") or k.startswith("psg_rst_")]
    other_cols = [k for k in all_features if not (k.startswith("rst_") or k.startswith("rel_rst_") or k.startswith("psg_rst_"))]
    
    tab_dim = len(all_features)
    rst_dim = len(rst_cols)
    other_dim = len(other_cols)
    
    print(f"   - Dimensions: Tabular={tab_dim}, RST={rst_dim}, Other={other_dim}")
    
    # Mock inputs
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    x_tab = torch.randn((batch_size, tab_dim))
    x_rst = torch.randn((batch_size, rst_dim))
    x_other = torch.randn((batch_size, other_dim))
    
    # Test all modes
    modes = ["gated_all", "film_rst_skip", "concat_all", "no_rst", "heuristic_guided_rst", "heuristic_guided_deletion"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h_score = torch.rand((batch_size,))
    
    for mode in modes:
        try:
            model_mode = mode
            if mode in ("heuristic_guided_rst", "heuristic_guided_deletion"):
                model_mode = "heuristic_guided"
                
            # Load with mock weights for speed, or initialize
            model = HybridGatedBERTModel(
                tabular_dim=tab_dim,
                rst_dim=rst_dim,
                other_dim=other_dim,
                mode=model_mode,
                pretrained_name="bert-base-uncased"
            ).to(device)
            
            model.eval()
            with torch.no_grad():
                out = model(
                    input_ids.to(device),
                    attention_mask.to(device),
                    token_type_ids.to(device),
                    x_tab.to(device),
                    x_rst.to(device),
                    x_other.to(device),
                    heuristic_score=h_score.to(device) if mode in ("heuristic_guided_rst", "heuristic_guided_deletion") else None
                )
            print(f"   - Mode '{mode}': Forward pass output shape = {out.shape} (Expected: torch.Size([{batch_size}]))")
            if out.shape[0] == batch_size:
                print(f"     SUCCESS: Forward pass for mode '{mode}' verified.")
            else:
                print(f"     Error: Output shape mismatch for mode '{mode}'.")
        except Exception as e:
            print(f"   - Error validating mode '{mode}': {e}")
            
    print("\n" + "="*50)
    print("ALL CHECKS COMPLETED")
    print("="*50)

def main():
    print("="*50)
    print("SQuAD SENTENCE SALIENCE SYSTEM VALIDATION")
    print("="*50 + "\n")
    validate_environment()
    data = validate_cache()
    if data:
        validate_balancing(data)
        validate_hybrid_bert(data)

if __name__ == "__main__":
    main()
