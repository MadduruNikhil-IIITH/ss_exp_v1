import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class SurprisalCalculator:
    def __init__(self, model_name="gpt2", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading surprisal calculator with model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure offset mapping is supported (standard for GPT-2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_token_surprisals(self, text):
        """
        Runs the text through the causal LM and computes token surprisals in bits: -log2 P(w_i | w_<i).
        Returns a list of dictionaries with token text, character spans, and surprisal value.
        """
        if not text.strip():
            return []
            
        inputs = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        offset_mapping = inputs["offset_mapping"][0].numpy()
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # Shape: (1, seq_len, vocab_size)
            
        # Shift logits and input_ids to align predictions with targets
        shift_logits = logits[0, :-1, :]  # Shape: (seq_len - 1, vocab_size)
        shift_labels = input_ids[0, 1:]   # Shape: (seq_len - 1)
        
        # Log-probability calculation
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs[range(len(shift_labels)), shift_labels]
        
        # Convert to surprisal in bits: -log2(p)
        surprisals = -target_log_probs / np.log(2.0)
        surprisals = surprisals.cpu().numpy()
        
        # The first token has no history context, assign 0.0 or the mean as default
        all_surprisals = [0.0] + list(surprisals)
        
        token_data = []
        for i, token_id in enumerate(input_ids[0]):
            tok_text = self.tokenizer.decode([token_id])
            start, end = offset_mapping[i]
            token_data.append({
                "text": tok_text,
                "start": start,
                "end": end,
                "surprisal": float(all_surprisals[i])
            })
            
        return token_data

    def extract_surprisal_features(self, token_data, sentence_boundaries):
        """
        Maps context token surprisals to sentence boundaries and computes multi-level features:
        - Sentence-level stats (mean, max, min, sum, std)
        - Passage-level baseline stats
        - Relative features (sentence compared to passage baseline)
        """
        if not token_data or not sentence_boundaries:
            return [{} for _ in sentence_boundaries]
            
        # 1. Passage-level stats
        passage_surprisals = [t["surprisal"] for t in token_data]
        passage_mean = float(np.mean(passage_surprisals))
        passage_sum = float(np.sum(passage_surprisals))
        passage_max = float(np.max(passage_surprisals))
        passage_min = float(np.min(passage_surprisals))
        passage_std = float(np.std(passage_surprisals))
        
        sentence_features = []
        
        for sent in sentence_boundaries:
            s_start = sent["start_char"]
            s_end = sent["end_char"]
            
            # Map tokens to this sentence
            # We select tokens whose start_char lies inside the sentence span
            sent_tokens = [t for t in token_data if s_start <= t["start"] < s_end]
            
            if not sent_tokens:
                # Default empty features
                sentence_features.append({
                    "surp_mean": 0.0, "surp_max": 0.0, "surp_min": 0.0, "surp_sum": 0.0, "surp_std": 0.0,
                    "psg_surp_mean": passage_mean, "psg_surp_sum": passage_sum,
                    "rel_surp_diff": 0.0, "rel_surp_ratio": 1.0, "rel_surp_sum_ratio": 0.0
                })
                continue
                
            sent_surprisals = [t["surprisal"] for t in sent_tokens]
            s_mean = float(np.mean(sent_surprisals))
            s_max = float(np.max(sent_surprisals))
            s_min = float(np.min(sent_surprisals))
            s_sum = float(np.sum(sent_surprisals))
            s_std = float(np.std(sent_surprisals)) if len(sent_surprisals) > 1 else 0.0
            
            # Relative features
            rel_diff = s_mean - passage_mean
            rel_ratio = s_mean / max(1e-8, passage_mean)
            rel_sum_ratio = s_sum / max(1e-8, passage_sum)
            
            sentence_features.append({
                # Sentence level
                "surp_mean": s_mean,
                "surp_max": s_max,
                "surp_min": s_min,
                "surp_sum": s_sum,
                "surp_std": s_std,
                # Passage level baseline
                "psg_surp_mean": passage_mean,
                "psg_surp_sum": passage_sum,
                "psg_surp_max": passage_max,
                "psg_surp_min": passage_min,
                "psg_surp_std": passage_std,
                # Relative
                "rel_surp_diff": rel_diff,
                "rel_surp_ratio": rel_ratio,
                "rel_surp_sum_ratio": rel_sum_ratio
            })
            
        return sentence_features
