import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm
from src.classifiers.rule_based_rst import RuleBasedRSTClassifier

class SQuADPairwiseDataset(Dataset):
    """
    PyTorch Dataset for pairwise SQuAD sentence salience.
    Prepares tokenized pairs for sentence 1 and sentence 2, along with their tabular features.
    """
    def __init__(self, records, tokenizer, feature_columns, rst_columns, other_columns, max_len=128, heuristic_mode=None):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.heuristic_mode = heuristic_mode
        
        self.feature_columns = feature_columns
        self.rst_columns = rst_columns
        self.other_columns = other_columns
        
        if heuristic_mode == "rst":
            self._rst_scorer = RuleBasedRSTClassifier()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        question = record["question"]
        s1_text = record["s1_text"]
        s2_text = record["s2_text"]
        
        # Tokenize pair 1: (Question, s1_text)
        enc1 = self.tokenizer(
            question,
            s1_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        # Tokenize pair 2: (Question, s2_text)
        enc2 = self.tokenizer(
            question,
            s2_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        # Tabular features vectors for s1
        feats1 = record["s1_features"]
        x_tab1 = torch.tensor([feats1.get(c, 0.0) for c in self.feature_columns], dtype=torch.float)
        x_rst1 = torch.tensor([feats1.get(c, 0.0) for c in self.rst_columns], dtype=torch.float)
        x_other1 = torch.tensor([feats1.get(c, 0.0) for c in self.other_columns], dtype=torch.float)
        
        # Tabular features vectors for s2
        feats2 = record["s2_features"]
        x_tab2 = torch.tensor([feats2.get(c, 0.0) for c in self.feature_columns], dtype=torch.float)
        x_rst2 = torch.tensor([feats2.get(c, 0.0) for c in self.rst_columns], dtype=torch.float)
        x_other2 = torch.tensor([feats2.get(c, 0.0) for c in self.other_columns], dtype=torch.float)
        
        y = torch.tensor(record["label"], dtype=torch.float)
        
        result = {
            "input_ids1": enc1["input_ids"].flatten(),
            "attention_mask1": enc1["attention_mask"].flatten(),
            "token_type_ids1": enc1.get("token_type_ids", torch.zeros_like(enc1["input_ids"])).flatten(),
            "x_tab1": x_tab1,
            "x_rst1": x_rst1,
            "x_other1": x_other1,
            
            "input_ids2": enc2["input_ids"].flatten(),
            "attention_mask2": enc2["attention_mask"].flatten(),
            "token_type_ids2": enc2.get("token_type_ids", torch.zeros_like(enc2["input_ids"])).flatten(),
            "x_tab2": x_tab2,
            "x_rst2": x_rst2,
            "x_other2": x_other2,
            
            "label": y
        }
        
        if self.heuristic_mode == "rst":
            h1 = self._rst_scorer.score_sentence(record["s1_features"])
            h2 = self._rst_scorer.score_sentence(record["s2_features"])
            result["heuristic_score1"] = torch.tensor(h1, dtype=torch.float)
            result["heuristic_score2"] = torch.tensor(h2, dtype=torch.float)
        elif self.heuristic_mode == "deletion":
            h1 = record["s1_features"].get("surp_deletion_drop", 0.0)
            h2 = record["s2_features"].get("surp_deletion_drop", 0.0)
            result["heuristic_score1"] = torch.tensor(h1, dtype=torch.float)
            result["heuristic_score2"] = torch.tensor(h2, dtype=torch.float)
        
        return result

class SQuADSentenceDataset(Dataset):
    """
    PyTorch Dataset for SQuAD sentence salience.
    Prepares text tokens and matches tabular features.
    """
    def __init__(self, records, tokenizer, feature_columns, rst_columns, other_columns, max_len=128, use_soft_labels=False, heuristic_mode=None):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_soft_labels = use_soft_labels
        self.heuristic_mode = heuristic_mode
        
        self.feature_columns = feature_columns
        self.rst_columns = rst_columns
        self.other_columns = other_columns
        
        if heuristic_mode == "rst":
            self._rst_scorer = RuleBasedRSTClassifier()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        question = record["question"]
        sentence = record["sentence_text"]
        
        # Tokenize pair: (Question, Sentence)
        encoding = self.tokenizer(
            question,
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        # Tabular features vectors
        feats = record["features"]
        x_tab = torch.tensor([feats.get(c, 0.0) for c in self.feature_columns], dtype=torch.float)
        x_rst = torch.tensor([feats.get(c, 0.0) for c in self.rst_columns], dtype=torch.float)
        x_other = torch.tensor([feats.get(c, 0.0) for c in self.other_columns], dtype=torch.float)
        
        # Targets
        if self.use_soft_labels:
            y = torch.tensor(record.get("soft_label_hybrid", record["binary_label"]), dtype=torch.float)
        else:
            y = torch.tensor(record["binary_label"], dtype=torch.float)
            
        result = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).flatten(),
            "x_tab": x_tab,
            "x_rst": x_rst,
            "x_other": x_other,
            "label": y
        }
        
        if self.heuristic_mode == "rst":
            h = self._rst_scorer.score_sentence(record["features"])
            result["heuristic_score"] = torch.tensor(h, dtype=torch.float)
        elif self.heuristic_mode == "deletion":
            h = record["features"].get("surp_deletion_drop", 0.0)
            result["heuristic_score"] = torch.tensor(h, dtype=torch.float)
        
        return result

class HybridGatedBERTModel(nn.Module):
    """
    Custom Hybrid BERT network combining BERT text embeddings with tabular features.
    Supports Gated, Concatenated, and Forced RST Integration (FiLM + Skip Connections).
    """
    def __init__(self, tabular_dim, rst_dim, other_dim, mode="film_rst_skip", pretrained_name="bert-base-uncased"):
        super().__init__()
        self.mode = mode
        self.bert = BertModel.from_pretrained(pretrained_name)
        bert_dim = self.bert.config.hidden_size  # 768
        
        self.tabular_dim = tabular_dim
        self.rst_dim = rst_dim
        self.other_dim = other_dim
        
        # 1. Feature Projection Layers
        if mode == "gated_all" or mode == "concat_all":
            self.proj_tab = nn.Sequential(
                nn.Linear(tabular_dim, bert_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        elif mode == "no_rst":
            self.proj_tab = nn.Sequential(
                nn.Linear(other_dim, bert_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        elif mode == "film_rst_skip":
            # FiLM parameter generators from RST features (outputs scale and shift for BERT dimensions)
            self.film_gamma = nn.Linear(rst_dim, bert_dim)
            self.film_beta = nn.Linear(rst_dim, bert_dim)
        elif mode == "heuristic_guided":
            # Single learnable scalar weight that scales the heuristic score
            self.heuristic_weight = nn.Linear(1, 1, bias=True)
            
        # 2. Fusion / Gating / Classification Head
        if mode == "gated_all" or mode == "no_rst":
            # Learn vector gate combining BERT embedding and projected tabular features
            self.gate_layer = nn.Sequential(
                nn.Linear(bert_dim * 2, bert_dim),
                nn.Sigmoid()
            )
            self.classifier = nn.Linear(bert_dim, 1)
        elif mode == "concat_all":
            self.classifier = nn.Linear(bert_dim + bert_dim, 1)
        elif mode == "film_rst_skip":
            # Skip link:modulated BERT embedding (768) + raw RST features + raw other features
            # Direct link prevents the optimizer from ignoring tabular features.
            self.classifier = nn.Linear(bert_dim + rst_dim + other_dim, 1)
        elif mode == "heuristic_guided":
            # BERT text logit fused with a single learned heuristic contribution
            self.classifier = nn.Linear(bert_dim, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, x_tab, x_rst, x_other, heuristic_score=None):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Pooler output (CLS token representation)
        h_bert = bert_outputs.pooler_output  # Shape: (batch_size, 768)
        
        if self.mode == "gated_all":
            h_tab = self.proj_tab(x_tab)
            gate = self.gate_layer(torch.cat([h_bert, h_tab], dim=-1))
            h_combined = gate * h_bert + (1.0 - gate) * h_tab
            logits = self.classifier(h_combined)
            
        elif self.mode == "no_rst":
            # x_other contains linguistic + surprisal + alignment features
            h_other = self.proj_tab(x_other)
            gate = self.gate_layer(torch.cat([h_bert, h_other], dim=-1))
            h_combined = gate * h_bert + (1.0 - gate) * h_other
            logits = self.classifier(h_combined)
            
        elif self.mode == "concat_all":
            h_tab = self.proj_tab(x_tab)
            h_combined = torch.cat([h_bert, h_tab], dim=-1)
            logits = self.classifier(h_combined)
            
        elif self.mode == "film_rst_skip":
            # FiLM scaling and shifting from RST features
            gamma = self.film_gamma(x_rst)
            beta = self.film_beta(x_rst)
            h_modulated = gamma * h_bert + beta
            
            # Concatenate modulated BERT vector with raw RST and other tabular features directly
            h_combined = torch.cat([h_modulated, x_rst, x_other], dim=-1)
            logits = self.classifier(h_combined)
        
        elif self.mode == "heuristic_guided":
            # BERT text logit + learned scalar contribution from the RST heuristic
            # logits = W_bert^T h_bert + b_bert + W_h * h + b_h
            bert_logit = self.classifier(h_bert)  # Shape: (batch_size, 1)
            # heuristic_score: (batch_size,) → (batch_size, 1)
            h_score = heuristic_score.unsqueeze(-1)
            h_contribution = self.heuristic_weight(h_score)  # Shape: (batch_size, 1)
            logits = (bert_logit + h_contribution).squeeze(-1)
            return logits
            
        return logits.squeeze(-1)

class HybridBERTClassifier:
    """
    Wrapper class managing dataset prep, training, and predicting with Hybrid BERT.
    """
    def __init__(self, mode="film_rst_skip", pretrained_name="bert-base-uncased", device="cuda"):
        self.mode = mode
        self.pretrained_name = pretrained_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        self.model = None
        
        # Columns setup
        self.feature_cols = []
        self.rst_cols = []
        self.other_cols = []

    def _setup_columns(self, all_feature_keys):
        from src.data_processing import classify_feature_keys
        categories = classify_feature_keys(all_feature_keys)
        self.feature_cols = categories["all"]
        self.rst_cols = categories["rst"]
        # Other cols: linguistic + surprisal + alignment
        self.other_cols = categories["linguistic"] + categories["surprisal"] + categories["alignment"]

    def fit(self, train_records, val_records=None, epochs=3, batch_size=8, lr=2e-5, use_soft_labels=False):
        """
        Fits the Hybrid Gated/FiLM BERT model.
        """
        is_pairwise = "s1_features" in train_records[0]
        
        if is_pairwise:
            all_feature_keys = train_records[0]["s1_features"].keys()
        else:
            all_feature_keys = train_records[0]["features"].keys()
            
        self._setup_columns(all_feature_keys)
        
        # Determine heuristic mode and model mode
        is_heuristic = None
        model_mode = self.mode
        if self.mode in ("heuristic_guided_rst", "heuristic_guided"):
            is_heuristic = "rst"
            model_mode = "heuristic_guided"
        elif self.mode == "heuristic_guided_deletion":
            is_heuristic = "deletion"
            model_mode = "heuristic_guided"
            
        # Datasets & Dataloaders
        if is_pairwise:
            train_dataset = SQuADPairwiseDataset(
                train_records, self.tokenizer, self.feature_cols, self.rst_cols, self.other_cols, heuristic_mode=is_heuristic
            )
        else:
            train_dataset = SQuADSentenceDataset(
                train_records, self.tokenizer, self.feature_cols, self.rst_cols, self.other_cols, use_soft_labels=use_soft_labels, heuristic_mode=is_heuristic
            )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize network
        tab_dim = len(self.feature_cols)
        rst_dim = len(self.rst_cols)
        other_dim = len(self.other_cols)
        
        self.model = HybridGatedBERTModel(
            tabular_dim=tab_dim,
            rst_dim=rst_dim,
            other_dim=other_dim,
            mode=model_mode,
            pretrained_name=self.pretrained_name
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        # Use BCEWithLogitsLoss for binary targets (both standard and pairwise), and MSELoss for continuous targets
        criterion = nn.MSELoss() if use_soft_labels else nn.BCEWithLogitsLoss()
        
        print(f"Training HybridGatedBERT ({self.mode} mode) on {self.device} for {epochs} epochs...")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                if is_pairwise:
                    input_ids1 = batch["input_ids1"].to(self.device)
                    attention_mask1 = batch["attention_mask1"].to(self.device)
                    token_type_ids1 = batch["token_type_ids1"].to(self.device)
                    x_tab1 = batch["x_tab1"].to(self.device)
                    x_rst1 = batch["x_rst1"].to(self.device)
                    x_other1 = batch["x_other1"].to(self.device)
                    
                    input_ids2 = batch["input_ids2"].to(self.device)
                    attention_mask2 = batch["attention_mask2"].to(self.device)
                    token_type_ids2 = batch["token_type_ids2"].to(self.device)
                    x_tab2 = batch["x_tab2"].to(self.device)
                    x_rst2 = batch["x_rst2"].to(self.device)
                    x_other2 = batch["x_other2"].to(self.device)
                    
                    labels = batch["label"].to(self.device)
                    
                    h_score1 = batch["heuristic_score1"].to(self.device) if is_heuristic else None
                    h_score2 = batch["heuristic_score2"].to(self.device) if is_heuristic else None
                    logits1 = self.model(input_ids1, attention_mask1, token_type_ids1, x_tab1, x_rst1, x_other1, heuristic_score=h_score1)
                    logits2 = self.model(input_ids2, attention_mask2, token_type_ids2, x_tab2, x_rst2, x_other2, heuristic_score=h_score2)
                    
                    # Pairwise difference: l_diff = logits1 - logits2
                    loss = criterion(logits1 - logits2, labels)
                else:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    token_type_ids = batch["token_type_ids"].to(self.device)
                    x_tab = batch["x_tab"].to(self.device)
                    x_rst = batch["x_rst"].to(self.device)
                    x_other = batch["x_other"].to(self.device)
                    labels = batch["label"].to(self.device)
                    h_score = batch["heuristic_score"].to(self.device) if is_heuristic else None
                    
                    logits = self.model(input_ids, attention_mask, token_type_ids, x_tab, x_rst, x_other, heuristic_score=h_score)
                    loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            print(f"Epoch {epoch+1} Complete. Train Loss: {train_loss / len(train_loader):.4f}")
            
            # Validation step if available (validation records are always single-sentence)
            if val_records:
                val_loss = self.evaluate_loss(val_records, batch_size, criterion, use_soft_labels=False)
                print(f"Validation Loss: {val_loss:.4f}")
                
        return self

    def evaluate_loss(self, val_records, batch_size, criterion, use_soft_labels):
        self.model.eval()
        is_heuristic = None
        if self.mode in ("heuristic_guided_rst", "heuristic_guided"):
            is_heuristic = "rst"
        elif self.mode == "heuristic_guided_deletion":
            is_heuristic = "deletion"
            
        val_dataset = SQuADSentenceDataset(
            val_records, self.tokenizer, self.feature_cols, self.rst_cols, self.other_cols,
            use_soft_labels=use_soft_labels, heuristic_mode=is_heuristic
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                x_tab = batch["x_tab"].to(self.device)
                x_rst = batch["x_rst"].to(self.device)
                x_other = batch["x_other"].to(self.device)
                labels = batch["label"].to(self.device)
                h_score = batch["heuristic_score"].to(self.device) if is_heuristic else None
                
                logits = self.model(input_ids, attention_mask, token_type_ids, x_tab, x_rst, x_other, heuristic_score=h_score)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
        return val_loss / len(val_loader)

    def predict_proba(self, test_records, batch_size=8):
        """
        Predicts probabilities/scores for a set of test records.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        
        is_heuristic = None
        if self.mode in ("heuristic_guided_rst", "heuristic_guided"):
            is_heuristic = "rst"
        elif self.mode == "heuristic_guided_deletion":
            is_heuristic = "deletion"
            
        self.model.eval()
        test_dataset = SQuADSentenceDataset(
            test_records, self.tokenizer, self.feature_cols, self.rst_cols, self.other_cols,
            use_soft_labels=False, heuristic_mode=is_heuristic
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        scores = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                x_tab = batch["x_tab"].to(self.device)
                x_rst = batch["x_rst"].to(self.device)
                x_other = batch["x_other"].to(self.device)
                h_score = batch["heuristic_score"].to(self.device) if is_heuristic else None
                
                logits = self.model(input_ids, attention_mask, token_type_ids, x_tab, x_rst, x_other, heuristic_score=h_score)
                # Apply sigmoid squash to get probability scores
                probas = torch.sigmoid(logits).cpu().numpy()
                scores.extend(probas)
                
        return np.array(scores)

    def predict(self, test_records, threshold=0.5, batch_size=8):
        """
        Predicts binary labels (0 or 1).
        """
        probas = self.predict_proba(test_records, batch_size)
        return (probas >= threshold).astype(int)
