import numpy as np

class RuleBasedRSTClassifier:
    """
    Heuristic rule-based classifier that assigns a salience score to a sentence
    based purely on its Rhetorical Structure Theory (RST) features.
    """
    def __init__(self, weight_root=2.0, weight_n_ratio=1.5, weight_contrast=1.0, weight_cause=1.0, weight_depth=-0.5):
        self.weight_root = weight_root
        self.weight_n_ratio = weight_n_ratio
        self.weight_contrast = weight_contrast
        self.weight_cause = weight_cause
        self.weight_depth = weight_depth

    def score_sentence(self, feats):
        """
        Computes a heuristic salience score. Higher score = more salient.
        """
        score = 0.0
        
        # 1. Root presence (strong indicator of centrality)
        score += self.weight_root * feats.get("rst_is_root", 0.0)
        
        # 2. Nucleus density (centrality of clauses inside the sentence)
        score += self.weight_n_ratio * feats.get("rst_n_ratio", 0.5)
        
        # 3. High-salience relation markers
        contrast_count = feats.get("rst_rel_contrast_count", 0.0)
        score += self.weight_contrast * contrast_count
        
        cause_count = feats.get("rst_rel_cause_count", 0.0)
        score += self.weight_cause * cause_count
        
        # 4. Depth penalty (nested discourse units represent minor details)
        rel_depth = feats.get("rel_rst_depth_ratio", 0.0)
        score += self.weight_depth * rel_depth
        
        # Sigmoid squash to project scores between 0 and 1
        return 1.0 / (1.0 + np.exp(-score))

    def predict_proba(self, X_feats):
        """
        X_feats can be a list of feature dictionaries.
        Returns an array of squashed salience scores (probabilities).
        """
        scores = []
        for feats in X_feats:
            scores.append(self.score_sentence(feats))
        return np.array(scores)

    def predict(self, X_feats, threshold=0.5):
        """
        Returns binary predictions (0 or 1).
        """
        probas = self.predict_proba(X_feats)
        return (probas >= threshold).astype(int)
