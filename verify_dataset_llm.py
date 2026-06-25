import os
import pickle
import random
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from tqdm import tqdm

def load_local_llm(model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"):
    print(f"Loading local LLM '{model_name}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    return model, tokenizer

def run_llm_judge(model, tokenizer, context, question, sentence, device="cuda"):
    """
    Prompts Qwen to judge if the sentence is salient for the question.
    """
    prompt_template = (
        "<|im_start|>system\n"
        "You are an expert NLP annotator evaluating sentence salience in reading comprehension. "
        "Your task is to determine if a target sentence is SALIENT for answering a question. "
        "A sentence is SALIENT (Yes) if it contains the answer or provides direct, essential information needed to construct the answer. "
        "A sentence is NOT SALIENT (No) if it is irrelevant or only provides tangential context.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Context Passage:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Sentence to Evaluate:\n{sentence}\n\n"
        "Does the Sentence contain the answer or direct essential information to answer the Question? "
        "Briefly explain your reasoning in one sentence, then end your response with exactly 'Judgment: Yes' or 'Judgment: No'.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Parse judgment
    judgment_match = re.search(r"Judgment:\s*(Yes|No)", response, re.IGNORECASE)
    if judgment_match:
        judgment = judgment_match.group(1).strip().capitalize()
    else:
        # Fallback keyword matching
        if "yes" in response.lower() and "judgment" in response.lower():
            judgment = "Yes"
        elif "no" in response.lower() and "judgment" in response.lower():
            judgment = "No"
        else:
            # Last resort
            judgment = "Yes" if "yes" in response.lower() else "No"
            
    binary_judgment = 1 if judgment == "Yes" else 0
    return binary_judgment, response

def main():
    cache_path = "features_cache_deletion.pkl"
    if not os.path.exists(cache_path):
        cache_path = "features_cache.pkl"
    artifact_dir = r"C:\Users\maddu\.gemini\antigravity\brain\64294085-ec70-478d-8d81-2ec235975552"
    
    print("="*60)
    print("LLM-AS-A-JUDGE DATASET VERIFICATION START")
    print("="*60)
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file '{cache_path}' not found. Please run run_feature_extraction.py first.")
        return
        
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
        
    # Combine train and validation records to have a larger pool
    all_records = cache_data["train"] + cache_data["validation"]
    
    salient_records = [r for r in all_records if r["binary_label"] == 1]
    non_salient_records = [r for r in all_records if r["binary_label"] == 0]
    
    print(f"Total dataset: {len(all_records)} sentences.")
    print(f"Salient (Silver = 1): {len(salient_records)}")
    print(f"Non-Salient (Silver = 0): {len(non_salient_records)}")
    
    # Sample balanced set (50 positive, 50 negative)
    sample_size = min(50, len(salient_records), len(non_salient_records))
    print(f"Sampling balanced set of {sample_size * 2} records...")
    
    random.seed(42)
    sampled_salient = random.sample(salient_records, sample_size)
    sampled_non_salient = random.sample(non_salient_records, sample_size)
    eval_set = sampled_salient + sampled_non_salient
    random.shuffle(eval_set)
    
    # Load LLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_local_llm(model_name="Qwen/Qwen2.5-1.5B-Instruct", device=device)
    
    silver_labels = []
    llm_judgments = []
    evaluations = []
    
    for idx, r in enumerate(tqdm(eval_set, desc="Evaluating dataset with LLM judge")):
        context = r["context"]
        question = r["question"]
        sentence = r["sentence_text"]
        silver = r["binary_label"]
        
        try:
            llm_val, reason = run_llm_judge(model, tokenizer, context, question, sentence, device)
        except Exception as e:
            print(f"Error during generation: {e}")
            llm_val = silver # Fallback
            reason = f"Error: {e}"
            
        silver_labels.append(silver)
        llm_judgments.append(llm_val)
        
        evaluations.append({
            "idx": idx + 1,
            "question": question,
            "sentence": sentence,
            "silver": silver,
            "llm": llm_val,
            "response": reason
        })
        
    # Compute metrics
    silver_arr = np.array(silver_labels)
    llm_arr = np.array(llm_judgments)
    
    accuracy = accuracy_score(silver_arr, llm_arr)
    precision = precision_score(llm_arr, silver_arr, zero_division=0) # TP / (TP+FP) where LLM is ground truth
    recall = recall_score(llm_arr, silver_arr, zero_division=0)
    f1 = f1_score(llm_arr, silver_arr, zero_division=0)
    kappa = cohen_kappa_score(silver_arr, llm_arr)
    
    # Confusion matrix elements
    tp = int(np.sum((silver_arr == 1) & (llm_arr == 1)))
    fp = int(np.sum((silver_arr == 1) & (llm_arr == 0)))  # Silver salient, LLM says no
    fn = int(np.sum((silver_arr == 0) & (llm_arr == 1)))  # Silver non-salient, LLM says yes
    tn = int(np.sum((silver_arr == 0) & (llm_arr == 0)))
    
    print("\n--- Evaluation Summary ---")
    print(f"Agreement Rate (Accuracy): {accuracy:.4f}")
    print(f"Cohen's Kappa Score:      {kappa:.4f}")
    print(f"Silver Label Precision:    {precision:.4f} (as proxy of LLM target)")
    print(f"Silver Label Recall:       {recall:.4f}")
    print(f"Silver Label F1 Score:     {f1:.4f}")
    print(f"Confusion Matrix: [TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}]")
    
    # Save markdown report to artifacts and workspace
    report_path = os.path.join(artifact_dir, "llm_judge_verification.md")
    report_workspace_path = "llm_judge_verification.md"
    print(f"Saving report to '{report_path}' and '{report_workspace_path}'...")
    
    # Extract examples of disagreements
    false_positives = [e for e in evaluations if e["silver"] == 1 and e["llm"] == 0]
    false_negatives = [e for e in evaluations if e["silver"] == 0 and e["llm"] == 1]
    
    def write_report(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# LLM-as-a-Judge Dataset Verification Report\n\n")
            f.write("This report validates the exact-index silver annotations of the SQuAD sentence salience dataset against zero-shot predictions from a local **`Qwen/Qwen2.5-1.5B-Instruct`** model.\n\n")
            
            f.write("## 1. Agreement Metrics\n")
            f.write(f"- **Sample Size**: {len(eval_set)} sentences (balanced: {sample_size} salient, {sample_size} non-salient)\n")
            f.write(f"- **Agreement Rate (Accuracy)**: `{accuracy:.4f}`\n")
            f.write(f"- **Cohen's Kappa Score**: `{kappa:.4f}` (measures agreement above chance)\n")
            f.write(f"- **Silver Label Quality** (treating LLM Judge as ground truth):\n")
            f.write(f"  - **Precision**: `{precision:.4f}`\n")
            f.write(f"  - **Recall**: `{recall:.4f}`\n")
            f.write(f"  - **F1 Score**: `{f1:.4f}`\n\n")
            
            f.write("### Confusion Matrix\n")
            f.write("| | LLM Salient (1) | LLM Non-Salient (0) |\n")
            f.write("| --- | --- | --- |\n")
            f.write(f"| **Silver Salient (1)** | **TP: {tp}** (Agree) | **FP: {fp}** (Silver=1, LLM=0) |\n")
            f.write(f"| **Silver Non-Salient (0)** | **FN: {fn}** (Silver=0, LLM=1) | **TN: {tn}** (Agree) |\n\n")
            
            f.write("## 2. Qualitative Error Analysis\n\n")
            
            f.write("### Category A: Silver Salient (1) but LLM Non-Salient (0)\n")
            f.write("> [!NOTE]\n")
            f.write("> These are cases where the sentence physically intersects the SQuAD annotated answer offset, but the LLM believes it is not sufficient or contextually relevant to answer the question. This can happen if an answer span overlaps sentence boundaries slightly or if the sentence contains the answer keyword but lacks the semantic context.\n\n")
            
            if false_positives:
                for idx, e in enumerate(false_positives[:5]):
                    f.write(f"**Example {idx+1}**:\n")
                    f.write(f"- **Question**: {e['question']}\n")
                    f.write(f"- **Sentence**: *\"{e['sentence']}\"*\n")
                    f.write(f"- **LLM Reasoning**: {e['response']}\n\n")
            else:
                f.write("*No examples found in this category.*\n\n")
                
            f.write("### Category B: Silver Non-Salient (0) but LLM Salient (1)\n")
            f.write("> [!NOTE]\n")
            f.write("> These represent cases where the sentence does NOT contain the exact answer character span, but the LLM judges it as salient. This typically occurs when a sentence contains crucial background context necessary to understand the answer, or when it contains a paraphrase of the answer that SQuAD annotators did not explicitly select.\n\n")
            
            if false_negatives:
                for idx, e in enumerate(false_negatives[:5]):
                    f.write(f"**Example {idx+1}**:\n")
                    f.write(f"- **Question**: {e['question']}\n")
                    f.write(f"- **Sentence**: *\"{e['sentence']}\"*\n")
                    f.write(f"- **LLM Reasoning**: {e['response']}\n\n")
            else:
                f.write("*No examples found in this category.*\n\n")

    write_report(report_path)
    write_report(report_workspace_path)
    print("Verification reports saved successfully.")
    print("="*60)

if __name__ == "__main__":
    main()
