import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

class DiscourseQGPipeline:
    """
    Discourse-Aware Question Generation (QG) and Closed-Loop QA Agent Verification Pipeline.
    Supports batched generation for fast execution on large datasets.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        if self.model is None:
            print(f"Loading local LLM '{self.model_name}' on {self.device} for QG & QA Agent...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            ).to(self.device)
            self.model.eval()

    def generate_question(self, context, salient_sentence):
        """
        Generates a reading comprehension question based on context and target sentence (Single instance).
        """
        self._load_model()
        return self.generate_questions_batch([context], [salient_sentence], batch_size=1)[0]

    def answer_question(self, context, question):
        """
        QA Agent solver: attempts to answer a question given context (Single instance).
        """
        self._load_model()
        return self.answer_questions_batch([context], [question], batch_size=1)[0]

    def generate_questions_batch(self, contexts, salient_sentences, batch_size=16):
        """
        Generates questions in batches to maximize GPU performance.
        """
        self._load_model()
        prompts = []
        for ctx, sent in zip(contexts, salient_sentences):
            prompt = (
                "<|im_start|>system\n"
                "You are a professional reading comprehension question creator. "
                "Your task is to write a single, natural, clear reading comprehension question "
                "based on the context passage, such that the target sentence is the exact and complete answer to the question. "
                "Do not reference the target sentence explicitly in your question.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"Context Passage:\n{ctx}\n\n"
                f"Target Sentence (Answer): {sent}\n\n"
                "Generate the question. Output ONLY the question text.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            prompts.append(prompt)
            
        questions = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="QG Batches"):
            batch_prompts = prompts[i:i+batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            for j, out in enumerate(outputs):
                input_len = inputs.input_ids[j].shape[-1]
                generated_ids = out[input_len:]
                q = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                q = re.sub(r'^["\']|["\']$', '', q).strip()
                questions.append(q)
        return questions

    def answer_questions_batch(self, contexts, questions, batch_size=16):
        """
        QA Agent solver: attempts to answer questions in batches.
        """
        self._load_model()
        prompts = []
        for ctx, q in zip(contexts, questions):
            prompt = (
                "<|im_start|>system\n"
                "You are an expert reading comprehension solver. "
                "Given the context passage, answer the user's question. "
                "Provide a short, direct answer extracted from the context.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"Context Passage:\n{ctx}\n\n"
                f"Question:\n{q}\n\n"
                "Provide the answer. Output ONLY the answer text.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            prompts.append(prompt)
            
        answers = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="QA Batches"):
            batch_prompts = prompts[i:i+batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            for j, out in enumerate(outputs):
                input_len = inputs.input_ids[j].shape[-1]
                generated_ids = out[input_len:]
                ans = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                answers.append(ans)
        return answers

    def verify_question(self, target_sentence, predicted_answer):
        """
        Verifies if the QA agent's predicted answer matches or overlaps significantly with the target sentence.
        """
        target_clean = re.sub(r'[^\w\s]', '', target_sentence.lower()).strip()
        pred_clean = re.sub(r'[^\w\s]', '', predicted_answer.lower()).strip()
        
        if pred_clean in target_clean or target_clean in pred_clean:
            return True
            
        target_tokens = set(target_clean.split())
        pred_tokens = set(pred_clean.split())
        intersection = target_tokens.intersection(pred_tokens)
        
        min_len = min(len(target_tokens), len(pred_tokens))
        if min_len > 0 and len(intersection) / min_len >= 0.3:
            return True
            
        return False
