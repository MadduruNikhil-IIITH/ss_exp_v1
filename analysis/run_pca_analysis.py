import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_pca_diagnostic(cache_path="../features_cache.pkl"):
    print("="*80)
    print("RUNNING PCA DIAGNOSTIC PIPELINE")
    print("="*80)
    
    # Resolve relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_cache_path = os.path.abspath(os.path.join(script_dir, cache_path))
    
    if not os.path.exists(abs_cache_path):
        print(f"Error: Cache file '{abs_cache_path}' not found. Please run Stage 2 first.")
        return
        
    with open(abs_cache_path, "rb") as f:
        cache_data = pickle.load(f)
        
    train_records = cache_data["train"]
    val_records = cache_data["validation"]
    
    # 1. Extract feature names and tabular arrays
    feature_keys = sorted(list(train_records[0]["features"].keys()))
    print(f"Loaded {len(feature_keys)} features from cache.")
    
    X_train = np.array([[r["features"][k] for k in feature_keys] for r in train_records])
    y_train = np.array([r["binary_label"] for r in train_records])
    
    X_val = np.array([[r["features"][k] for k in feature_keys] for r in val_records])
    y_val = np.array([r["binary_label"] for r in val_records])
    
    print(f"Train feature shape: {X_train.shape}")
    print(f"Val feature shape:   {X_val.shape}")
    
    # 2. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 3. Fit PCA
    pca = PCA()
    pca.fit(X_train_scaled)
    
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    
    # Find components explaining 90%, 95%, 99% variance
    comp_90 = np.argmax(cum_var >= 0.90) + 1
    comp_95 = np.argmax(cum_var >= 0.95) + 1
    comp_99 = np.argmax(cum_var >= 0.99) + 1
    
    print(f"\n--- PCA Variance Contribution ---")
    print(f"Components explaining 90% variance: {comp_90} / {len(feature_keys)}")
    print(f"Components explaining 95% variance: {comp_95} / {len(feature_keys)}")
    print(f"Components explaining 99% variance: {comp_99} / {len(feature_keys)}")
    
    # 4. Generate & Save Elbow Curve Plot
    docs_dir = os.path.abspath(os.path.join(script_dir, "../docs/images"))
    os.makedirs(docs_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='-', color='b', label='Cumulative Variance')
    plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.5, align='center', color='g', label='Individual Variance')
    
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
    plt.axvline(x=comp_95, color='r', linestyle=':', label=f'{comp_95} Components')
    
    plt.title("PCA Explained Variance Elbow Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Principal Components", fontsize=12)
    plt.ylabel("Explained Variance Ratio", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plot_path = os.path.join(docs_dir, "pca_elbow_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"PCA elbow curve saved to '{plot_path}'")
    
    # 5. Evaluate pre-PCA Logistic Regression
    print(f"\n--- Model Comparison: Raw Scaling vs PCA Reduction (95% Variance) ---")
    
    lr_raw = LogisticRegression(max_iter=1000, random_state=42)
    lr_raw.fit(X_train_scaled, y_train)
    y_pred_raw = lr_raw.predict(X_val_scaled)
    
    raw_acc = accuracy_score(y_val, y_pred_raw)
    raw_prec = precision_score(y_val, y_pred_raw, zero_division=0)
    raw_rec = recall_score(y_val, y_pred_raw, zero_division=0)
    raw_f1 = f1_score(y_val, y_pred_raw, zero_division=0)
    
    print(f"Raw Features (Scaled, {len(feature_keys)} dims):")
    print(f"  Accuracy:  {raw_acc:.4f}")
    print(f"  Precision: {raw_prec:.4f}")
    print(f"  Recall:    {raw_rec:.4f}")
    print(f"  F1 Score:  {raw_f1:.4f}")
    
    # 6. Evaluate post-PCA Logistic Regression (using comp_95)
    pca_reduced = PCA(n_components=comp_95)
    X_train_pca = pca_reduced.fit_transform(X_train_scaled)
    X_val_pca = pca_reduced.transform(X_val_scaled)
    
    lr_pca = LogisticRegression(max_iter=1000, random_state=42)
    lr_pca.fit(X_train_pca, y_train)
    y_pred_pca = lr_pca.predict(X_val_pca)
    
    pca_acc = accuracy_score(y_val, y_pred_pca)
    pca_prec = precision_score(y_val, y_pred_pca, zero_division=0)
    pca_rec = recall_score(y_val, y_pred_pca, zero_division=0)
    pca_f1 = f1_score(y_val, y_pred_pca, zero_division=0)
    
    print(f"\nPCA Features ({comp_95} components):")
    print(f"  Accuracy:  {pca_acc:.4f}")
    print(f"  Precision: {pca_prec:.4f}")
    print(f"  Recall:    {pca_rec:.4f}")
    print(f"  F1 Score:  {pca_f1:.4f}")
    print("="*80)

if __name__ == "__main__":
    run_pca_diagnostic()
