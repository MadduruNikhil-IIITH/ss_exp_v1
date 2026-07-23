import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm
import math
from src.classifiers.focal_loss import BinaryFocalLoss

class SQuADSequenceDataset(Dataset):
    """
    Sequence-level PyTorch Dataset for SQuAD sentence salience.
    Groups sentence-question records by question_id and sorts them by sentence_idx.
    Returns full sequences representing entire contexts.
    """
    def __init__(self, records, tokenizer, feature_columns, rst_columns, other_columns, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.feature_columns = feature_columns
        self.rst_columns = rst_columns
        self.other_columns = other_columns
        
        # Group records by question_id
        grouped = {}
        for r in records:
            q_id = r["question_id"]
            if q_id not in grouped:
                grouped[q_id] = []
            grouped[q_id].append(r)
            
        # Sort each group by sentence_idx and store as sequence
        self.sequences = []
        for q_id, group in grouped.items():
            sorted_group = sorted(group, key=lambda x: x["sentence_idx"])
            self.sequences.append(sorted_group)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Tokenize each sentence in the sequence
        seq_input_ids = []
        seq_attention_mask = []
        seq_token_type_ids = []
        
        for r in seq:
            text = r["sentence_text"]
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            seq_input_ids.append(encoding["input_ids"].flatten())
            seq_attention_mask.append(encoding["attention_mask"].flatten())
            seq_token_type_ids.append(encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).flatten())
            
        # Convert lists to tensors
        input_ids = torch.stack(seq_input_ids)
        attention_mask = torch.stack(seq_attention_mask)
        token_type_ids = torch.stack(seq_token_type_ids)
        
        # Tabular features lists
        x_tab = torch.tensor([[r["features"].get(c, 0.0) for c in self.feature_columns] for r in seq], dtype=torch.float)
        x_rst = torch.tensor([[r["features"].get(c, 0.0) for c in self.rst_columns] for r in seq], dtype=torch.float)
        x_other = torch.tensor([[r["features"].get(c, 0.0) for c in self.other_columns] for r in seq], dtype=torch.float)
        
        # Labels
        labels = torch.tensor([r["binary_label"] for r in seq], dtype=torch.float)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "x_tab": x_tab,
            "x_rst": x_rst,
            "x_other": x_other,
            "labels": labels,
            "seq_len": len(seq)
        }

def sequence_collate_fn(batch):
    """
    Custom collate function to pad variable-length sentence sequences in a batch.
    """
    max_seq_len = max(item["seq_len"] for item in batch)
    max_token_len = batch[0]["input_ids"].shape[1]
    
    tab_dim = batch[0]["x_tab"].shape[1]
    rst_dim = batch[0]["x_rst"].shape[1]
    other_dim = batch[0]["x_other"].shape[1]
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    padded_input_ids = torch.zeros((batch_size, max_seq_len, max_token_len), dtype=torch.long)
    padded_attention_mask = torch.zeros((batch_size, max_seq_len, max_token_len), dtype=torch.long)
    padded_token_type_ids = torch.zeros((batch_size, max_seq_len, max_token_len), dtype=torch.long)
    
    padded_x_tab = torch.zeros((batch_size, max_seq_len, tab_dim), dtype=torch.float)
    padded_x_rst = torch.zeros((batch_size, max_seq_len, rst_dim), dtype=torch.float)
    padded_x_other = torch.zeros((batch_size, max_seq_len, other_dim), dtype=torch.float)
    
    padded_labels = torch.full((batch_size, max_seq_len), -100.0, dtype=torch.float) # -100 for ignored elements
    seq_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
    
    for i, item in enumerate(batch):
        length = item["seq_len"]
        
        padded_input_ids[i, :length] = item["input_ids"]
        padded_attention_mask[i, :length] = item["attention_mask"]
        padded_token_type_ids[i, :length] = item["token_type_ids"]
        
        padded_x_tab[i, :length] = item["x_tab"]
        padded_x_rst[i, :length] = item["x_rst"]
        padded_x_other[i, :length] = item["x_other"]
        
        padded_labels[i, :length] = item["labels"]
        seq_mask[i, :length] = 1.0
        
    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "token_type_ids": padded_token_type_ids,
        "x_tab": padded_x_tab,
        "x_rst": padded_x_rst,
        "x_other": padded_x_other,
        "labels": padded_labels,
        "seq_mask": seq_mask
    }

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encodings for the sequence sequence model.
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x

class LinguisticallyGroundedSaliencyModel(nn.Module):
    """
    Linguistically-Grounded Saliency Model (LGSM) implementation:
    - Semantic Stream: Frozen BERT text representations.
    - Linguistic Stream: 1-layer MLP feature projection.
    - Adaptive Gated Fusion: Scalar gate dynamically weighting streams.
    - 2-Layer Transformer Encoder: Modeling context sequential dependencies.
    """
    def __init__(self, tabular_dim, rst_dim, other_dim, pretrained_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        
        # Freeze BERT weights to establish a strict semantic baseline
        for param in self.bert.parameters():
            param.requires_grad = False
            
        bert_dim = self.bert.config.hidden_size  # 768
        
        # 1. Linguistic Stream: project explicit tabular features
        self.proj_ling = nn.Sequential(
            nn.Linear(tabular_dim, bert_dim),
            nn.LayerNorm(bert_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Projection for gate matching
        self.proj_ling_fused = nn.Linear(bert_dim, bert_dim)
        
        # 2. Gated Fusion: learns scalar gating coefficient alpha_t
        self.gate_layer = nn.Sequential(
            nn.Linear(bert_dim + bert_dim, 1),
            nn.Sigmoid()
        )
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(bert_dim)
        
        # 4. Contextual sequence model: 2-layer Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bert_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim, 1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, x_tab, x_rst, x_other, seq_mask=None):
        # input_ids: [batch_size, seq_len, token_len]
        batch_size, seq_len, token_len = input_ids.shape
        
        # Reshape to run BERT on all sentences in batch sequentially
        flat_input_ids = input_ids.view(batch_size * seq_len, token_len)
        flat_attention_mask = attention_mask.view(batch_size * seq_len, token_len)
        flat_token_type_ids = token_type_ids.view(batch_size * seq_len, token_len)
        
        # Get semantic embeddings from CLS token
        bert_outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )
        h_sem = bert_outputs.pooler_output  # [batch_size * seq_len, 768]
        h_sem = h_sem.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 768]
        
        # Project explicit features (x_tab contains all tabular features)
        flat_x_tab = x_tab.view(batch_size * seq_len, -1)
        h_ling = self.proj_ling(flat_x_tab)  # [batch_size * seq_len, 768]
        h_ling = h_ling.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 768]
        
        # Compute scalar gating coefficient alpha_t for each sentence
        # Concatenate semantic and linguistic vectors along last dimension
        combined_streams = torch.cat([h_sem, h_ling], dim=-1)  # [batch_size, seq_len, 1536]
        alpha = self.gate_layer(combined_streams)  # [batch_size, seq_len, 1]
        
        # Project h_ling to same latent space
        h_ling_proj = self.proj_ling_fused(h_ling)
        
        # Convex Gated Fusion combination
        h_fused = alpha * h_ling_proj + (1.0 - alpha) * h_sem  # [batch_size, seq_len, 768]
        
        # Apply Positional Encoding
        h_fused = self.pos_encoder(h_fused)
        
        # Contextualize sequence through 2-layer Transformer
        # Transformer mask to prevent attending to padded padding items
        src_key_padding_mask = (seq_mask == 0.0) if seq_mask is not None else None
        h_context = self.transformer_encoder(h_fused, src_key_padding_mask=src_key_padding_mask)  # [batch_size, seq_len, 768]
        
        # Classifier output
        logits = self.classifier(h_context)  # [batch_size, seq_len, 1]
        
        return logits.squeeze(-1), alpha.squeeze(-1)

class LGSMSaliencyClassifier:
    """
    High-level Wrapper class to train, evaluate, and predict with the LGSM model.
    """
    def __init__(self, pretrained_name="bert-base-uncased", device="cuda"):
        self.pretrained_name = pretrained_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        self.model = None
        
        self.feature_cols = []
        self.rst_cols = []
        self.other_cols = []

    def _setup_columns(self, all_feature_keys, use_rst=True):
        from src.data_processing import classify_feature_keys
        categories = classify_feature_keys(all_feature_keys)
        self.feature_cols = categories["all"]
        self.rst_cols = categories["rst"]
        # Other cols: linguistic + surprisal + (no alignment keys)
        self.other_cols = [c for c in categories["linguistic"] + categories["surprisal"] if not c.startswith("align_")]
        
        if not use_rst:
            self.feature_cols = [c for c in self.feature_cols if not c.startswith("rst_") and not c.startswith("rel_rst_")]
            self.rst_cols = []
            self.other_cols = [c for c in self.other_cols if not c.startswith("rst_") and not c.startswith("rel_rst_")]
            
        self.tabular_dim = len(self.feature_cols)
        self.rst_dim = len(self.rst_cols)
        self.other_dim = len(self.other_cols)

    def fit(self, train_records, val_records=None, epochs=4, batch_size=4, lr=2e-5, use_rst=True):
        """
        Trains the LGSM sequence model using Focal Loss and AdamW.
        """
        all_feature_keys = train_records[0]["features"].keys()
        self._setup_columns(all_feature_keys, use_rst=use_rst)
        
        # Prepare datasets & loaders
        train_dataset = SQuADSequenceDataset(
            train_records, self.tokenizer, self.feature_cols, self.rst_cols, self.other_cols
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sequence_collate_fn)
        
        # Initialize LGSM model
        self.model = LinguisticallyGroundedSaliencyModel(
            tabular_dim=self.tabular_dim,
            rst_dim=self.rst_dim,
            other_dim=self.other_dim,
            pretrained_name=self.pretrained_name
        ).to(self.device)

        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        # Use custom Focal Loss
        criterion = BinaryFocalLoss(alpha=0.25, gamma=3.0, reduction='none')
        
        print(f"Training LGSM Model on {self.device} for {epochs} epochs...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_elements = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                x_tab = batch["x_tab"].to(self.device)
                x_rst = batch["x_rst"].to(self.device)
                x_other = batch["x_other"].to(self.device)
                labels = batch["labels"].to(self.device)
                seq_mask = batch["seq_mask"].to(self.device)
                
                # Forward
                logits, _ = self.model(input_ids, attention_mask, token_type_ids, x_tab, x_rst, x_other, seq_mask=seq_mask)
                
                # Compute loss (only on real items, mask out pad items)
                loss_elements = criterion(logits, labels)
                # Apply seq_mask
                masked_loss = loss_elements * seq_mask
                
                loss = masked_loss.sum() / max(seq_mask.sum(), 1.0)
                
                loss.backward()
                optimizer.step()
                
                total_loss += masked_loss.sum().item()
                total_elements += seq_mask.sum().item()
                
            print(f"Epoch {epoch+1} Loss: {total_loss / max(total_elements, 1.0):.4f}")
            
            # Run validation step
            if val_records:
                self.evaluate_val(val_records, criterion)
                
        return self

    def evaluate_val(self, val_records, criterion):
        self.model.eval()
        val_dataset = SQuADSequenceDataset(
            val_records, self.tokenizer, self.feature_cols, self.rst_cols, self.other_cols
        )
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=sequence_collate_fn)
        
        val_loss = 0.0
        val_elements = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                x_tab = batch["x_tab"].to(self.device)
                x_rst = batch["x_rst"].to(self.device)
                x_other = batch["x_other"].to(self.device)
                labels = batch["labels"].to(self.device)
                seq_mask = batch["seq_mask"].to(self.device)
                
                logits, _ = self.model(input_ids, attention_mask, token_type_ids, x_tab, x_rst, x_other, seq_mask=seq_mask)
                loss_elements = criterion(logits, labels)
                masked_loss = loss_elements * seq_mask
                val_loss += masked_loss.sum().item()
                val_elements += seq_mask.sum().item()
                
        print(f"Validation Loss: {val_loss / max(val_elements, 1.0):.4f}")

    def predict_proba(self, test_records, batch_size=4):
        """
        Predicts probabilities for a list of test records, keeping sequence alignment.
        """
        if self.model is None:
            raise ValueError("LGSM Model not trained yet!")
            
        self.model.eval()
        test_dataset = SQuADSequenceDataset(
            test_records, self.tokenizer, self.feature_cols, self.rst_cols, self.other_cols
        )
        # Keep shuffle=False to align predictions back with original test records
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=sequence_collate_fn)
        
        # We need to map batch sequence predictions back to the individual test_records in order
        # Map grouped sequences
        q_ids_order = [seq[0]["question_id"] for seq in test_dataset.sequences]
        
        probs_dict = {}
        gates_dict = {}
        
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                x_tab = batch["x_tab"].to(self.device)
                x_rst = batch["x_rst"].to(self.device)
                x_other = batch["x_other"].to(self.device)
                seq_mask = batch["seq_mask"].to(self.device)
                
                logits, alphas = self.model(input_ids, attention_mask, token_type_ids, x_tab, x_rst, x_other, seq_mask=seq_mask)
                
                probas = torch.sigmoid(logits).cpu().numpy()
                alphas = alphas.cpu().numpy()
                
                batch_size_cur = input_ids.shape[0]
                for b in range(batch_size_cur):
                    global_seq_idx = idx * batch_size + b
                    if global_seq_idx >= len(q_ids_order):
                        break
                    q_id = q_ids_order[global_seq_idx]
                    
                    # Store probabilities and gate values, slice using seq_mask
                    real_len = int(seq_mask[b].sum().item())
                    probs_dict[q_id] = probas[b, :real_len]
                    gates_dict[q_id] = alphas[b, :real_len]
                    
        # Now, unpack in order of original flat test_records
        flat_probas = []
        flat_gates = []
        
        # Track our progress through each group
        progress_dict = {q_id: 0 for q_id in probs_dict.keys()}
        
        for r in test_records:
            q_id = r["question_id"]
            offset = progress_dict[q_id]
            
            flat_probas.append(probs_dict[q_id][offset])
            flat_gates.append(gates_dict[q_id][offset])
            
            progress_dict[q_id] += 1
            
        return np.array(flat_probas), np.array(flat_gates)

    def predict(self, test_records, threshold=0.5, batch_size=4):
        probas, _ = self.predict_proba(test_records, batch_size)
        return (probas >= threshold).astype(int)

    def save(self, filepath):
        """
        Saves the LGSM model checkpoint.
        """
        import torch
        checkpoint = {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "pretrained_name": self.pretrained_name,
            "feature_cols": self.feature_cols,
            "rst_cols": self.rst_cols,
            "other_cols": self.other_cols,
            "tabular_dim": self.tabular_dim,
            "rst_dim": self.rst_dim,
            "other_dim": self.other_dim
        }
        torch.save(checkpoint, filepath)
        print(f"Saved LGSMSaliencyClassifier checkpoint to '{filepath}'")

    def load(self, filepath, device="cuda"):
        """
        Loads the LGSM model checkpoint.
        """
        import torch
        checkpoint = torch.load(filepath, map_location=device)
        self.pretrained_name = checkpoint["pretrained_name"]
        self.feature_cols = checkpoint["feature_cols"]
        self.rst_cols = checkpoint["rst_cols"]
        self.other_cols = checkpoint["other_cols"]
        self.tabular_dim = checkpoint["tabular_dim"]
        self.rst_dim = checkpoint["rst_dim"]
        self.other_dim = checkpoint["other_dim"]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_name)
        
        self.model = LinguisticallyGroundedSaliencyModel(
            tabular_dim=self.tabular_dim,
            rst_dim=self.rst_dim,
            other_dim=self.other_dim,
            pretrained_name=self.pretrained_name
        ).to(self.device)
        
        if checkpoint["model_state_dict"] is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"Loaded LGSMSaliencyClassifier checkpoint from '{filepath}'")

