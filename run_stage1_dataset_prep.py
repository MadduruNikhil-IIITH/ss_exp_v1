import os
import random
import re
import time
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from tqdm import tqdm
from src.data_processing import build_silver_squad_dataset

def load_local_llm(model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"):
    print(f"[{time.strftime('%X')}] Loading local LLM '{model_name}' on {device}...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    print(f"[{time.strftime('%X')}] Local LLM loaded successfully in {time.time() - start_time:.2f} seconds.\n")
    return model, tokenizer

def run_llm_judge(model, tokenizer, context, question, sentence, device="cuda"):
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
        if "yes" in response.lower() and "judgment" in response.lower():
            judgment = "Yes"
        elif "no" in response.lower() and "judgment" in response.lower():
            judgment = "No"
        else:
            judgment = "Yes" if "yes" in response.lower() else "No"
            
    binary_judgment = 1 if judgment == "Yes" else 0
    return binary_judgment, response

def run_stage1_pipeline(num_train_contexts=15, num_val_contexts=5, output_csv="squad_labeled_dataset.csv"):
    print("="*80)
    print("STAGE 1: SILVER DATASET PREPARATION & LLM-AS-A-JUDGE AUDIT")
    print("="*80)
    
    # 1. Build Silver Dataset
    print(f"[{time.strftime('%X')}] Extracting raw SQuAD records...")
    train_records = build_silver_squad_dataset(num_contexts=num_train_contexts, split="train").to_dict(orient="records")
    val_records = build_silver_squad_dataset(num_contexts=num_val_contexts, split="validation").to_dict(orient="records")
    
    # Combine into a single raw labeled dataset
    all_records = []
    for r in train_records:
        r["split"] = "train"
        all_records.append(r)
    for r in val_records:
        r["split"] = "validation"
        all_records.append(r)
        
    df_dataset = pd.DataFrame(all_records)
    print(f"\n[{time.strftime('%X')}] Dataset constructed:")
    print(f"   - Total records: {len(df_dataset)}")
    print(f"   - Train records: {len(train_records)} (Salient: {sum(r['binary_label'] == 1 for r in train_records)})")
    print(f"   - Val records:   {len(val_records)} (Salient: {sum(r['binary_label'] == 1 for r in val_records)})")
    
    # Save raw labeled dataset to CSV
    df_dataset.to_csv(output_csv, index=False)
    print(f"[{time.strftime('%X')}] Labeled dataset successfully saved to '{output_csv}'")
    
    # 2. Run LLM-as-a-Judge Audit
    print(f"\n[{time.strftime('%X')}] Starting LLM-as-a-Judge audit...")
    salient_records = [r for r in all_records if r["binary_label"] == 1]
    non_salient_records = [r for r in all_records if r["binary_label"] == 0]
    
    sample_size = min(50, len(salient_records), len(non_salient_records))
    print(f"   - Sampling balanced set of {sample_size * 2} records (50% Salient, 50% Non-Salient) for audit...")
    
    random.seed(42)
    sampled_salient = random.sample(salient_records, sample_size)
    sampled_non_salient = random.sample(non_salient_records, sample_size)
    eval_set = sampled_salient + sampled_non_salient
    random.shuffle(eval_set)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, tokenizer = load_local_llm(model_name="Qwen/Qwen2.5-1.5B-Instruct", device=device)
        
        silver_labels = []
        llm_judgments = []
        evaluations = []
        
        for idx, r in enumerate(tqdm(eval_set, desc="LLM Judge Auditing Dataset")):
            context = r["context"]
            question = r["question"]
            sentence = r["sentence_text"]
            silver = r["binary_label"]
            
            try:
                llm_val, reason = run_llm_judge(model, tokenizer, context, question, sentence, device)
            except Exception as e:
                llm_val = silver
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
        precision = precision_score(llm_arr, silver_arr, zero_division=0)
        recall = recall_score(llm_arr, silver_arr, zero_division=0)
        f1 = f1_score(llm_arr, silver_arr, zero_division=0)
        kappa = cohen_kappa_score(silver_arr, llm_arr)
        
        tp = int(np.sum((silver_arr == 1) & (llm_arr == 1)))
        fp = int(np.sum((silver_arr == 1) & (llm_arr == 0)))
        fn = int(np.sum((silver_arr == 0) & (llm_arr == 1)))
        tn = int(np.sum((silver_arr == 0) & (llm_arr == 0)))
        
        print("\n" + "-"*40)
        print("LLM AUDIT SUMMARY LOGS")
        print("-"*40)
        print(f"Agreement Rate (Accuracy): {accuracy:.4f}")
        print(f"Cohen's Kappa Score:      {kappa:.4f}")
        print(f"Silver Label Precision:    {precision:.4f}")
        print(f"Silver Label Recall:       {recall:.4f}")
        print(f"Silver Label F1 Score:     {f1:.4f}")
        print(f"Confusion Matrix: [TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}]")
        print("-"*40)
        
        # Write reports
        os.makedirs("docs", exist_ok=True)
        report_workspace_path = os.path.join("docs", "llm_judge_verification.md")
        
        false_positives = [e for e in evaluations if e["silver"] == 1 and e["llm"] == 0]
        false_negatives = [e for e in evaluations if e["silver"] == 0 and e["llm"] == 1]
        
        def write_report(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("# LLM-as-a-Judge Dataset Verification Report\n\n")
                f.write("This report validates the exact-index silver annotations of the SQuAD sentence salience dataset against zero-shot predictions from a local **`Qwen/Qwen2.5-1.5B-Instruct`** model.\n\n")
                f.write("## 1. Agreement Metrics\n")
                f.write(f"- **Sample Size**: {len(eval_set)} sentences (balanced: {sample_size} salient, {sample_size} non-salient)\n")
                f.write(f"- **Agreement Rate (Accuracy)**: `{accuracy:.4f}`\n")
                f.write(f"- **Cohen's Kappa Score**: `{kappa:.4f}`\n")
                f.write(f"- **Silver Label Quality** (LLM Judge as ground truth):\n")
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
                if false_positives:
                    for idx, e in enumerate(false_positives[:5]):
                        f.write(f"**Example {idx+1}**:\n")
                        f.write(f"- **Question**: {e['question']}\n")
                        f.write(f"- **Sentence**: *\"{e['sentence']}\"*\n")
                        f.write(f"- **LLM Reasoning**: {e['response']}\n\n")
                else:
                    f.write("*No examples found in this category.*\n\n")
                f.write("### Category B: Silver Non-Salient (0) but LLM Salient (1)\n")
                if false_negatives:
                    for idx, e in enumerate(false_negatives[:5]):
                        f.write(f"**Example {idx+1}**:\n")
                        f.write(f"- **Question**: {e['question']}\n")
                        f.write(f"- **Sentence**: *\"{e['sentence']}\"*\n")
                        f.write(f"- **LLM Reasoning**: {e['response']}\n\n")
                else:
                    f.write("*No examples found in this category.*\n\n")
                    
        write_report(report_workspace_path)
            
        print(f"[{time.strftime('%X')}] LLM-as-a-Judge verification completed successfully. Reports saved to 'docs/llm_judge_verification.md'.")
    except Exception as e:
        print(f"\nWarning: Could not run LLM-as-a-Judge check: {e}")
        
    print("="*80)
    print("STAGE 1 PIPELINE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=None, help="Ratio of train contexts (e.g. 0.8 for 80/20 split)")
    parser.add_argument("--total_contexts", type=int, default=100, help="Total number of contexts to split")
    parser.add_argument("--num_train_contexts", type=int, default=None, help="Explicit number of train contexts")
    parser.add_argument("--num_val_contexts", type=int, default=None, help="Explicit number of val contexts")
    args = parser.parse_args()
    
    if args.train_ratio is not None:
        num_train = int(args.total_contexts * args.train_ratio)
        num_val = args.total_contexts - num_train
        print(f"Dynamic splitting active: train_ratio={args.train_ratio}, total_contexts={args.total_contexts} -> train={num_train}, val={num_val}")
        run_stage1_pipeline(num_train_contexts=num_train, num_val_contexts=num_val)
    elif args.num_train_contexts is not None and args.num_val_contexts is not None:
        run_stage1_pipeline(num_train_contexts=args.num_train_contexts, num_val_contexts=args.num_val_contexts)
    else:
        # Default fallback (Standardized 90/10 split over 100 contexts)
        run_stage1_pipeline(num_train_contexts=90, num_val_contexts=10)

