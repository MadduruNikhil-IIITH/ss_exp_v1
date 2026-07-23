import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from src.classifiers.rule_based_rst import RuleBasedRSTClassifier

class TabularClassifierWrapper:
    """
    Wrapper for training linear/logistic classifiers on extracted tabular features.
    Supports binary classification (Logistic Regression) and continuous targets (Ridge Regression).
    """
    def __init__(self, feature_mode="combined", use_soft_targets=False):
        self.feature_mode = feature_mode
        self.use_soft_targets = use_soft_targets
        self.scaler = StandardScaler()
        
        if use_soft_targets:
            self.model = Ridge(alpha=1.0)
        else:
            self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
            
        self.selected_cols = []

    def get_feature_columns(self, feature_keys):
        """
        Filters feature keys based on the feature_mode configuration.
        """
        from src.data_processing import classify_feature_keys
        categories = classify_feature_keys(feature_keys)
        
        if self.feature_mode == "rst":
            return categories["rst"]
        elif self.feature_mode == "linguistic":
            return categories["linguistic"]
        elif self.feature_mode == "surprisal":
            return categories["surprisal"]
        elif self.feature_mode == "alignment":
            return categories["alignment"]
        elif self.feature_mode == "combined":
            return categories["all"]
        elif self.feature_mode == "combined_no_rst":
            return [k for k in categories["all"] if not k.startswith("rst_") and not k.startswith("rel_rst_")]
        elif self.feature_mode == "combined_heuristic":
            return categories["all"]
        elif self.feature_mode == "combined_deletion":
            return [k for k in categories["all"] if k != "surp_deletion_drop"]
        else:
            raise ValueError(f"Unknown feature mode: {self.feature_mode}")

    def prepare_data(self, records, fit_scaler=True):
        """
        Converts list of records into X (scaled numpy array) and y.
        """
        if not records:
            return np.array([]), np.array([])
            
        # Detect if records are pairwise
        is_pairwise = "s1_features" in records[0]
        
        if is_pairwise:
            # Get feature keys from first record's s1_features
            all_feature_keys = records[0]["s1_features"].keys()
            self.selected_cols = self.get_feature_columns(all_feature_keys)
            
            X_raw = []
            y = []
            for r in records:
                feats1 = r["s1_features"]
                feats2 = r["s2_features"]
                diff = [feats1.get(c, 0.0) - feats2.get(c, 0.0) for c in self.selected_cols]
                X_raw.append(diff)
                y.append(r["label"])
            X_raw = np.array(X_raw)
            y = np.array(y)
            
            # --- combined_heuristic: append the difference of the RST rule-based scores ---
            if self.feature_mode == "combined_heuristic":
                rst_scorer = RuleBasedRSTClassifier()
                h1_scores = rst_scorer.predict_proba([r["s1_features"] for r in records])
                h2_scores = rst_scorer.predict_proba([r["s2_features"] for r in records])
                h_diff = (h1_scores - h2_scores).reshape(-1, 1)
                X_raw = np.hstack([X_raw, h_diff])
            elif self.feature_mode == "combined_deletion":
                h1_scores = np.array([r["s1_features"].get("surp_deletion_drop", 0.0) for r in records])
                h2_scores = np.array([r["s2_features"].get("surp_deletion_drop", 0.0) for r in records])
                h_diff = (h1_scores - h2_scores).reshape(-1, 1)
                X_raw = np.hstack([X_raw, h_diff])
        else:
            # Get feature keys from first record if training or selected_cols not set
            if fit_scaler or not self.selected_cols:
                all_feature_keys = records[0]["features"].keys()
                self.selected_cols = self.get_feature_columns(all_feature_keys)
            
            X_raw = []
            for r in records:
                feats = r["features"]
                X_raw.append([feats.get(c, 0.0) for c in self.selected_cols])
            X_raw = np.array(X_raw)
            
            # --- combined_heuristic: append the RST rule-based score as an extra column ---
            if self.feature_mode == "combined_heuristic":
                rst_scorer = RuleBasedRSTClassifier()
                heuristic_scores = rst_scorer.predict_proba([r["features"] for r in records])
                # Shape: (N,) → reshape to (N, 1) and append
                X_raw = np.hstack([X_raw, heuristic_scores.reshape(-1, 1)])
            elif self.feature_mode == "combined_deletion":
                deletion_scores = np.array([r["features"].get("surp_deletion_drop", 0.0) for r in records])
                X_raw = np.hstack([X_raw, deletion_scores.reshape(-1, 1)])
            
            # Select target
            if self.use_soft_targets:
                # Default to hybrid soft labels if using soft targets
                y = np.array([r.get("soft_label_hybrid", r["binary_label"]) for r in records])
            else:
                y = np.array([r["binary_label"] for r in records])
            
        # Scaling
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_raw)
        else:
            X_scaled = self.scaler.transform(X_raw)
            
        return X_scaled, y

    def fit(self, train_records):
        """
        Trains the model.
        """
        X, y = self.prepare_data(train_records, fit_scaler=True)
        if len(X) == 0:
            print("Error: Empty training data.")
            return self
            
        self.model.fit(X, y)
        return self

    def predict_proba(self, test_records):
        """
        Returns salience probability/score for each record.
        """
        X, _ = self.prepare_data(test_records, fit_scaler=False)
        if len(X) == 0:
            return np.array([])
            
        if self.use_soft_targets:
            # For Ridge Regression, we clip the output between [0, 1]
            preds = self.model.predict(X)
            return np.clip(preds, 0.0, 1.0)
        else:
            # For Logistic Regression, return the probability of class 1
            return self.model.predict_proba(X)[:, 1]

    def predict(self, test_records, threshold=0.5):
        """
        Returns binary predictions (0 or 1).
        """
        probas = self.predict_proba(test_records)
        return (probas >= threshold).astype(int)

    def save(self, filepath):
        """
        Saves model and scaler to a file.
        """
        import joblib
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "selected_cols": self.selected_cols,
            "feature_mode": self.feature_mode,
            "use_soft_targets": self.use_soft_targets
        }, filepath)
        print(f"Saved tabular model checkpoint to '{filepath}'")

    def load(self, filepath):
        """
        Loads model and scaler from a file.
        """
        import joblib
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.selected_cols = data["selected_cols"]
        self.feature_mode = data["feature_mode"]
        self.use_soft_targets = data["use_soft_targets"]
        print(f"Loaded tabular model checkpoint from '{filepath}'")

