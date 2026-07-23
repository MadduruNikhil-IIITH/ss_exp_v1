import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

class LLMJudgeClassifier:
    """
    Zero-shot LLM-as-a-Judge Classifier using a local instruction-tuned LLM.
    Acts as a strong baseline by judging sentence question-worthiness.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        if self.model is None:
            print(f"Loading local LLM '{self.model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            ).to(self.device)
            self.model.eval()
            
    def fit(self, train_records=None):
        # Zero-shot model, no training/fine-tuning needed
        pass
        
    def predict_proba(self, records, batch_size=1):
        self._load_model()
        probas = []
        for r in tqdm(records, desc="LLM Judge Predicting"):
            context = r["context"]
            sentence = r["sentence_text"]
            
            prompt_template = (
                "<|im_start|>system\n"
                "You are an expert NLP annotator evaluating sentence salience in reading comprehension. "
                "Your task is to determine if a target sentence is QUESTION-WORTHY (salient). "
                "A sentence is SALIENT (Yes) if it contains key, central information, facts, or events in the context "
                "that are highly suitable for generating a reading comprehension question. "
                "A sentence is NOT SALIENT (No) if it contains minor details, background explanation, or is tangential.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"Context Passage:\n{context}\n\n"
                f"Sentence to Evaluate:\n{sentence}\n\n"
                "Is the Sentence salient and question-worthy within the Context? "
                "Briefly explain your reasoning in one sentence, then end your response with exactly 'Judgment: Yes' or 'Judgment: No'.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            inputs = self.tokenizer(prompt_template, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Parse Judgment
            judgment_match = re.search(r"Judgment:\s*(Yes|No)", response, re.IGNORECASE)
            if judgment_match:
                judgment = judgment_match.group(1).strip().capitalize()
            else:
                if "yes" in response.lower() and "judgment" in response.lower():
                    judgment = "Yes"
                elif "no" in response.lower() and "judgment" in response.lower():
                    judgment = "No"
                else:
                    judgment = "Yes" if "yes" in response.lower() else "No"
                    
            prob = 1.0 if judgment == "Yes" else 0.0
            probas.append(prob)
            
        return np.array(probas)
        
    def predict(self, records, threshold=0.5, batch_size=1):
        probas = self.predict_proba(records)
        return (probas >= threshold).astype(int)
        
    def save(self, filepath):
        # No weights to save
        pass
        
    def load(self, filepath, device="cuda"):
        # Loaded dynamically from AutoModel
        pass
