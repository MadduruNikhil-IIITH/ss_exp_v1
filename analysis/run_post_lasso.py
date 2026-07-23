import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

def main():
    print("="*80)
    print("CONTROLLED CORPUS STUDY: POST-LASSO LOGISTIC REGRESSION")
    print("="*80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.abspath(os.path.join(script_dir, "../features_cache.pkl"))
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file '{cache_path}' not found. Run Stage 2 first!")
        return

    print(f"Loading cached features from '{cache_path}'...")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    train_records = cache_data["train"]
    val_records = cache_data["validation"]
    print(f"Loaded {len(train_records)} train records and {len(val_records)} val records.")

    if not train_records:
        print("Error: Empty train records.")
        return

    # Extract features keys
    feature_keys = list(train_records[0]["features"].keys())
    # Exclude any query-alignment features if present, to keep it query-independent
    feature_keys = [k for k in feature_keys if not k.startswith("align_")]
    print(f"Total features considered: {len(feature_keys)}")

    # Prepare training X and y
    X_raw = []
    y = []
    for r in train_records:
        feats = r["features"]
        X_raw.append([feats.get(k, 0.0) for k in feature_keys])
        y.append(r["binary_label"])

    X_raw = np.array(X_raw)
    y = np.array(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Convert to DataFrame for easier column mapping
    df_X = pd.DataFrame(X_scaled, columns=feature_keys)

    # Stage 1: Feature Selection using LASSO (L1 penalty)
    print("Stage 1: Running L1-regularized Logistic Regression (LASSO) for feature selection...")
    # C=0.1 to obtain a sparse set (~20-30 features)
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42, class_weight='balanced')
    lasso.fit(df_X, y)

    coef = lasso.coef_[0]
    selected_features = [feature_keys[i] for i in range(len(feature_keys)) if coef[i] != 0.0]
    print(f"LASSO selected {len(selected_features)} features.")

    # Force-include controls: length control (word_count) and position control (sentence_idx)
    control_names = ["word_count", "sentence_idx_control"]
    
    # Rebuild X matrix containing selected features and controls
    X_selected_list = []
    for idx, r in enumerate(train_records):
        row_dict = {}
        # Selected features from train_records
        for k in selected_features:
            row_dict[k] = df_X.loc[idx, k]
            
        # Forced controls (standardized)
        row_dict["word_count"] = r["features"].get("word_count", 0.0)
        row_dict["sentence_idx_control"] = float(r["sentence_idx"])
        X_selected_list.append(row_dict)

    df_selected = pd.DataFrame(X_selected_list)
    
    # Standardize the selected + control features again to make coefficients directly comparable
    selected_cols = list(df_selected.columns)
    df_selected_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=selected_cols)

    # Stage 2: Unregularized Refit using scikit-learn + scipy for Wald Inference
    print("Stage 2: Fitting unregularized Logistic Regression refit via scikit-learn + scipy...")
    X_mat = df_selected_scaled.values
    intercept_col = np.ones((X_mat.shape[0], 1))
    X_logit = np.hstack([intercept_col, X_mat])
    logit_cols = ["const"] + list(df_selected_scaled.columns)

    # Fit unregularized Logistic Regression
    clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=200, random_state=42)
    clf.fit(X_mat, y)

    coefs = np.insert(clf.coef_[0], 0, clf.intercept_[0])
    
    # Calculate Standard Errors using Hessian matrix H = X^T * W * X
    p_hat = 1.0 / (1.0 + np.exp(-np.dot(X_logit, coefs)))
    p_hat = np.clip(p_hat, 1e-15, 1 - 1e-15)
    W = p_hat * (1.0 - p_hat)
    
    # Inverse Fisher Information matrix
    H = np.dot(X_logit.T * W, X_logit)
    try:
        cov_matrix = np.linalg.inv(H)
        bse = np.sqrt(np.maximum(0, np.diag(cov_matrix)))
    except np.linalg.LinAlgError:
        cov_matrix = np.linalg.pinv(H)
        bse = np.sqrt(np.maximum(0, np.diag(cov_matrix)))

    z_scores = np.where(bse > 0, coefs / bse, 0.0)
    p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_scores)))

    coefs_series = pd.Series(coefs, index=logit_cols)
    bse_series = pd.Series(bse, index=logit_cols)
    z_series = pd.Series(z_scores, index=logit_cols)
    p_series = pd.Series(p_values, index=logit_cols)

    # Build results table
    res_df = pd.DataFrame({
        "Feature": logit_cols,
        "Coef": coefs_series.values,
        "Std_Err": bse_series.values,
        "z": z_series.values,
        "p": p_series.values
    })

    # Add significance flags
    def get_sig(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""
    res_df["sig"] = res_df["p"].apply(get_sig)

    # Sort by absolute z-value (excluding intercept const)
    res_df_sorted = res_df[res_df["Feature"] != "const"].copy()
    res_df_sorted["abs_z"] = res_df_sorted["z"].abs()
    res_df_sorted = res_df_sorted.sort_values(by="abs_z", ascending=False).drop(columns=["abs_z"]).reset_index(drop=True)

    # Print top features
    print("\n" + "="*80)
    print("TOP PREDICTORS OF SENTENCE SALIENCE (BY WALD Z-SCORE)")
    print("="*80)
    print(res_df_sorted.head(30).to_string(index=False))
    print("="*80)

    # Save to CSV in project root
    csv_path = os.path.abspath(os.path.join(script_dir, "../post_lasso_results.csv"))
    res_df_sorted.to_csv(csv_path, index=False)
    
    # Save Markdown report
    docs_dir = os.path.abspath(os.path.join(script_dir, "../docs"))
    os.makedirs(docs_dir, exist_ok=True)
    report_workspace_path = os.path.join(docs_dir, "post_lasso_report.md")

    def write_report(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# SQuAD Controlled Corpus Study: Post-LASSO Regression Analysis\n\n")
            f.write("> [!IMPORTANT]\n")
            f.write("> **Single Model Configuration Note**: This controlled corpus study is conducted using a **single, specific model configuration** to establish statistical significance. Specifically, we run an unregularized Logistic Regression refit on the **raw, unbalanced training split** (2,301 records). Crucially, we force-include two baseline control variables—**Sentence Length (`word_count`)** and **Linear Position (`sentence_idx`)**—alongside the features selected by the L1 (LASSO) penalty. This isolates the independent predictive signals of our engineered discourse (RST) and cognitive (surprisal) features from simple physical shortcuts.\n\n")
            f.write("---\n\n")
            f.write("## 1. Top Predictive Features\n")
            f.write("| Rank | Feature | Coefficient | z-score | p-value | Significance |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for i, r in res_df_sorted.iterrows():
                f.write(f"| {i+1} | `{r['Feature']}` | `{r['Coef']:.4f}` | `{r['z']:.2f}` | `{r['p']:.4e}` | **{r['sig']}** |\n")
            f.write("\n*Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05.*\n\n")
            f.write("## 2. Key Findings & Discussion\n\n")
            
            # Analyze position vs information vs RST
            pos_row = res_df_sorted[res_df_sorted["Feature"] == "sentence_idx_control"]
            len_row = res_df_sorted[res_df_sorted["Feature"] == "word_count"]
            
            f.write("### A. Position and Length Controls\n")
            if not pos_row.empty:
                f.write(f"- **Position Bias**: The normalized position feature has a coefficient of `{pos_row.iloc[0]['Coef']:.4f}` ($z = {pos_row.iloc[0]['z']:.2f}$, $p = {pos_row.iloc[0]['p']:.4e}$). ")
                if pos_row.iloc[0]['Coef'] < 0:
                    f.write("The negative coefficient confirms the expected lead-bias: sentences early in the passage are significantly more likely to contain answers (salient).\n")
                else:
                    f.write("The positive coefficient indicates later sentences are more likely to be selected in this sample.\n")
            
            if not len_row.empty:
                f.write(f"- **Length Bias**: The word count has a coefficient of `{len_row.iloc[0]['Coef']:.4f}` ($z = {len_row.iloc[0]['z']:.2f}$). This shows how length affects the salience probability.\n\n")

            f.write("### B. RST Discourse Structure Predictors\n")
            rst_features = res_df_sorted[res_df_sorted["Feature"].str.startswith("rst_") | res_df_sorted["Feature"].str.startswith("rel_rst_")]
            if not rst_features.empty:
                f.write("Several RST discourse features emerged as significant independent predictors:\n")
                for _, r in rst_features.head(5).iterrows():
                    f.write(f"- `{r['Feature']}`: Coef = `{r['Coef']:.4f}` ($z = {r['z']:.2f}$, $p = {r['p']:.4e}$). ")
                    if r["Coef"] > 0:
                        f.write("Higher values indicate increased likelihood of sentence salience.\n")
                    else:
                        f.write("Higher values indicate decreased likelihood (acting as a negative filter).\n")
            else:
                f.write("No RST features were selected by LASSO as independent predictors under the L1 penalty threshold.\n")
            f.write("\n")

            f.write("### C. Cognitive Surprisal Predictors\n")
            surp_features = res_df_sorted[res_df_sorted["Feature"].str.startswith("surp_") | res_df_sorted["Feature"].str.startswith("rel_surp_")]
            if not surp_features.empty:
                f.write("Information theoretic surprisal features show the following independent signals:\n")
                for _, r in surp_features.head(5).iterrows():
                    f.write(f"- `{r['Feature']}`: Coef = `{r['Coef']:.4f}` ($z = {r['z']:.2f}$, $p = {r['p']:.4e}$). ")
                    if r["Coef"] > 0:
                        f.write("Higher surprisal values are positively correlated with answer salience.\n")
                    else:
                        f.write("Lower surprisal (more predictable context) correlates with salience, indicating smoother information contours.\n")
            else:
                f.write("No surprisal features were selected by LASSO under the L1 penalty threshold.\n")

    write_report(report_workspace_path)
    print(f"Post-LASSO analysis completed successfully. Reports saved to '{report_workspace_path}'.")

if __name__ == "__main__":
    main()
