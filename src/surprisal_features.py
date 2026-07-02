import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import copy
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

class SurprisalCalculator:
    def __init__(self, causal_model_name="gpt2", masked_model_name="bert-base-uncased", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading Causal model '{causal_model_name}' and Masked model '{masked_model_name}' on {self.device}...")
        
        # 1. Causal setup (GPT-2)
        self.causal_tokenizer = AutoTokenizer.from_pretrained(causal_model_name)
        if self.causal_tokenizer.pad_token is None:
            self.causal_tokenizer.pad_token = self.causal_tokenizer.eos_token
        self.causal_model = AutoModelForCausalLM.from_pretrained(causal_model_name).to(self.device)
        self.causal_model.eval()
        
        # Add special StimulusMarker to causal tokenizer
        self.causal_tokenizer.add_tokens(["[!StimulusMarker!]", " [!StimulusMarker!]"])
        self.causal_model.resize_token_embeddings(len(self.causal_tokenizer))
        
        # 2. Masked setup (BERT)
        self.masked_tokenizer = AutoTokenizer.from_pretrained(masked_model_name)
        if not self.masked_tokenizer.bos_token and self.masked_tokenizer.cls_token:
            self.masked_tokenizer.bos_token = self.masked_tokenizer.cls_token
        if not self.masked_tokenizer.eos_token and self.masked_tokenizer.sep_token:
            self.masked_tokenizer.eos_token = self.masked_tokenizer.sep_token
            
        self.masked_model = AutoModelForMaskedLM.from_pretrained(masked_model_name).to(self.device)
        self.masked_model.eval()
        
        # Add special StimulusMarker to masked tokenizer
        self.masked_tokenizer.add_tokens(["[!StimulusMarker!]", " [!StimulusMarker!]"])
        self.masked_model.resize_token_embeddings(len(self.masked_tokenizer))

    def segment_context(self, tokenizer, preceding_text, target_text, following_text):
        """
        Segments the context into preceding, target, and following token IDs
        using explicit PsychFormers boundary markers.
        """
        stimulus_str = f"{preceding_text} * {target_text} * {following_text}"
        stimulus_spaces = stimulus_str.replace("*", "[!StimulusMarker!]")
        
        encoded = tokenizer.encode(stimulus_spaces)
        marker_id1 = tokenizer.encode("[!StimulusMarker!]")[-1]
        marker_id2 = tokenizer.encode(" [!StimulusMarker!]")[-1]
        
        marker_idxs = []
        for idx, tok_id in enumerate(encoded):
            if tok_id in (marker_id1, marker_id2):
                marker_idxs.append(idx)
                
        if len(marker_idxs) < 2:
            # Fallback if boundaries cannot be resolved
            bos_id = tokenizer.bos_token_id or (tokenizer.cls_token_id if hasattr(tokenizer, "cls_token_id") else None)
            return [bos_id or 0], tokenizer.encode(target_text), []
            
        idx1, idx2 = marker_idxs[0], marker_idxs[1]
        preceding_context = encoded[:idx1]
        
        # Ensure preceding_context is a list of integers
        if not isinstance(preceding_context, list):
            preceding_context = list(preceding_context)
            
        # Ensure it starts with BOS/CLS
        bos_id = tokenizer.bos_token_id or (tokenizer.cls_token_id if hasattr(tokenizer, "cls_token_id") else None)
        if len(preceding_context) == 0 or preceding_context[0] != bos_id:
            if bos_id is not None:
                preceding_context = [bos_id] + preceding_context
                
        target_words = encoded[idx1 + 1:idx2]
        following_words = encoded[idx2 + 1:]
        
        # Ensure outputs are list of ints
        return list(preceding_context), list(target_words), list(following_words)

    def get_causal_surprisals(self, preceding_context, target_words):
        """
        Calculates token-level causal surprisals using PsychFormers logic.
        """
        if not target_words:
            return []
            
        current_context = copy.deepcopy(preceding_context)
        probs_list = []
        for target in target_words:
            input_tensor = torch.LongTensor([current_context]).to(self.device)
            with torch.no_grad():
                logits = self.causal_model(input_tensor).logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            probability = probs[0, target]
            current_context.append(target)
            probs_list.append(max(float(probability.item()), 1e-12))
            
        return list(-np.log2(probs_list))

    def get_masked_pll(self, preceding_context, following_words, target_words):
        """
        Calculates token-level Pseudo-Log-Likelihood (PLL) values using PsychFormers logic.
        """
        if not target_words:
            return []
            
        current_context = copy.deepcopy(preceding_context)
        probs_list = []
        for target in target_words:
            context_plus_mask = current_context + [self.masked_tokenizer.mask_token_id]
            context_plus_mask = context_plus_mask + following_words
            
            eos_id = self.masked_tokenizer.eos_token_id or (self.masked_tokenizer.sep_token_id if hasattr(self.masked_tokenizer, "sep_token_id") else None)
            if eos_id is not None:
                model_input_list = context_plus_mask + [eos_id]
            else:
                model_input_list = context_plus_mask
                
            mask_idx = model_input_list.index(self.masked_tokenizer.mask_token_id)
            input_tensor = torch.LongTensor([model_input_list]).to(self.device)
            
            with torch.no_grad():
                logits = self.masked_model(input_tensor).logits[:, mask_idx, :]
            probs = F.softmax(logits, dim=-1)
            probability = probs[0, target]
            current_context.append(target)
            probs_list.append(max(float(probability.item()), 1e-12))
            
        return list(-np.log2(probs_list))

    def _get_passage_mean_surprisal(self, text):
        if not text.strip():
            return 0.0
        inputs = self.causal_tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] <= 1:
            return 0.0
        if input_ids.shape[1] > 1024:
            input_ids = input_ids[:, :1024]
        with torch.no_grad():
            outputs = self.causal_model(input_ids)
            logits = outputs.logits
        shift_logits = logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs[range(len(shift_labels)), shift_labels]
        surprisals = -target_log_probs / np.log(2.0)
        return float(torch.mean(surprisals).item())

    def extract_surprisal_features(self, sentence_boundaries, context_text):
        """
        Extracts new PsychFormers (Causal & Masked PLL) features and sentence deletion drop.
        """
        original_mean = self._get_passage_mean_surprisal(context_text)
        
        # 1. Extract PsychFormers lists for all sentences
        pf_causal_lists = []
        pf_pll_lists = []
        deletion_drops = []
        
        from tqdm import tqdm
        for sent in tqdm(sentence_boundaries, desc="PsychFormers Surprisal/PLL", leave=False):
            s_start = sent["start_char"]
            s_end = sent["end_char"]
            
            preceding_text = context_text[:s_start]
            target_text = context_text[s_start:s_end]
            following_text = context_text[s_end:]
            
            # Causal PsychFormers
            p_ctx_c, t_w_c, f_w_c = self.segment_context(self.causal_tokenizer, preceding_text, target_text, following_text)
            c_surps = self.get_causal_surprisals(p_ctx_c, t_w_c)
            pf_causal_lists.append(c_surps)
            
            # Masked PLL PsychFormers
            p_ctx_m, t_w_m, f_w_m = self.segment_context(self.masked_tokenizer, preceding_text, target_text, following_text)
            m_plls = self.get_masked_pll(p_ctx_m, f_w_m, t_w_m)
            pf_pll_lists.append(m_plls)
            
            # Compute Sentence Deletion Coherence Drop
            modified_text = preceding_text + following_text
            modified_mean = self._get_passage_mean_surprisal(modified_text)
            deletion_drops.append(modified_mean - original_mean)
            
        # 2. Compute passage baselines for PsychFormers features
        flat_pf_causal = [val for sublist in pf_causal_lists for val in sublist]
        flat_pf_pll = [val for sublist in pf_pll_lists for val in sublist]
        
        psg_c_mean = float(np.mean(flat_pf_causal)) if flat_pf_causal else 0.0
        psg_c_sum = float(np.sum(flat_pf_causal)) if flat_pf_causal else 0.0
        psg_c_max = float(np.max(flat_pf_causal)) if flat_pf_causal else 0.0
        psg_c_min = float(np.min(flat_pf_causal)) if flat_pf_causal else 0.0
        psg_c_std = float(np.std(flat_pf_causal)) if flat_pf_causal else 0.0
        
        psg_m_mean = float(np.mean(flat_pf_pll)) if flat_pf_pll else 0.0
        psg_m_sum = float(np.sum(flat_pf_pll)) if flat_pf_pll else 0.0
        psg_m_max = float(np.max(flat_pf_pll)) if flat_pf_pll else 0.0
        psg_m_min = float(np.min(flat_pf_pll)) if flat_pf_pll else 0.0
        psg_m_std = float(np.std(flat_pf_pll)) if flat_pf_pll else 0.0
        
        sentence_features = []
        
        for idx, sent in enumerate(sentence_boundaries):
            sent_idx = sent["sentence_idx"]
            
            # A. PsychFormers Causal features
            c_vals = pf_causal_lists[idx]
            if not c_vals:
                sc_mean, sc_max, sc_min, sc_sum, sc_std = 0.0, 0.0, 0.0, 0.0, 0.0
                rel_c_diff, rel_c_ratio, rel_c_sum_ratio = 0.0, 1.0, 0.0
            else:
                sc_mean = float(np.mean(c_vals))
                sc_max = float(np.max(c_vals))
                sc_min = float(np.min(c_vals))
                sc_sum = float(np.sum(c_vals))
                sc_std = float(np.std(c_vals)) if len(c_vals) > 1 else 0.0
                rel_c_diff = sc_mean - psg_c_mean
                rel_c_ratio = sc_mean / max(1e-8, psg_c_mean)
                rel_c_sum_ratio = sc_sum / max(1e-8, psg_c_sum)
                
            # B. PsychFormers Masked PLL features
            m_vals = pf_pll_lists[idx]
            if not m_vals:
                sm_mean, sm_max, sm_min, sm_sum, sm_std = 0.0, 0.0, 0.0, 0.0, 0.0
                rel_m_diff, rel_m_ratio, rel_m_sum_ratio = 0.0, 1.0, 0.0
            else:
                sm_mean = float(np.mean(m_vals))
                sm_max = float(np.max(m_vals))
                sm_min = float(np.min(m_vals))
                sm_sum = float(np.sum(m_vals))
                sm_std = float(np.std(m_vals)) if len(m_vals) > 1 else 0.0
                rel_m_diff = sm_mean - psg_m_mean
                rel_m_ratio = sm_mean / max(1e-8, psg_m_mean)
                rel_m_sum_ratio = sm_sum / max(1e-8, psg_m_sum)
                
            sentence_features.append({
                # PsychFormers Causal GPT-2 Surprisal
                "surp_causal_pf_mean": sc_mean,
                "surp_causal_pf_max": sc_max,
                "surp_causal_pf_min": sc_min,
                "surp_causal_pf_sum": sc_sum,
                "surp_causal_pf_std": sc_std,
                "rel_surp_causal_pf_diff": rel_c_diff,
                "rel_surp_causal_pf_ratio": rel_c_ratio,
                "rel_surp_causal_pf_sum_ratio": rel_c_sum_ratio,
                
                # PsychFormers Masked BERT PLL Coherence
                "surp_pll_pf_mean": sm_mean,
                "surp_pll_pf_max": sm_max,
                "surp_pll_pf_min": sm_min,
                "surp_pll_pf_sum": sm_sum,
                "surp_pll_pf_std": sm_std,
                "rel_surp_pll_pf_diff": rel_m_diff,
                "rel_surp_pll_pf_ratio": rel_m_ratio,
                "rel_surp_pll_pf_sum_ratio": rel_m_sum_ratio,
                
                # Sentence Deletion Coherence Drop
                "surp_deletion_drop": float(deletion_drops[idx]) if idx < len(deletion_drops) else 0.0
            })
            
        return sentence_features
