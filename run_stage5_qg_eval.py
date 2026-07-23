import os
import pickle
import time
import re
import string
from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from rouge_score import rouge_scorer
import bert_score

from src.qg_pipeline import DiscourseQGPipeline
from src.classifiers.llm_judge import LLMJudgeClassifier

def normalize_text(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(str(s))))

def compute_exact_match(target_text, pred_answer):
    return float(normalize_text(target_text) == normalize_text(pred_answer))

def compute_token_f1(target_text, pred_answer):
    gold_toks = normalize_text(target_text).split()
    pred_toks = normalize_text(pred_answer).split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)

def evaluate_question_metrics(gen_qs, ref_qs, target_sents, pred_answers, rouge_calc):
    """
    Computes ROUGE-L, BERTScore F1, QA-EM, QA-Consistency F1, and Answer Recovery F1.
    """
    rouge_l_scores = []
    for g_q, r_q in zip(gen_qs, ref_qs):
        # Handle multiple reference questions if split by '|'
        refs = [r.strip() for r in r_q.split('|') if r.strip()]
        if not refs:
            refs = [r_q]
        scores = [rouge_calc.score(ref, g_q)['rougeL'].fmeasure for ref in refs]
        rouge_l_scores.append(max(scores))
        
    print("Computing BERTScore F1 on generated questions...")
    # Compute BERTScore F1 against main reference question using BERT (bert-base-uncased)
    main_refs = [r.split('|')[0].strip() for r in ref_qs]
    P, R, F1 = bert_score.score(gen_qs, main_refs, lang="en", model_type="bert-base-uncased", verbose=False)
    bert_scores = F1.cpu().numpy().tolist()
    
    qa_em_scores = []
    qa_f1_scores = []
    answer_rec_scores = []
    
    for sent, ans, r_q in zip(target_sents, pred_answers, ref_qs):
        # 1. QA Consistency against target sentence
        em = compute_exact_match(sent, ans)
        f1 = compute_token_f1(sent, ans)
        qa_em_scores.append(em)
        qa_f1_scores.append(f1)
        
        # 2. Answer Recovery against target sentence/ground truth
        answer_rec_scores.append(f1)
        
    return {
        "rouge_l": float(np.mean(rouge_l_scores)),
        "bert_score_f1": float(np.mean(bert_scores)),
        "qa_em": float(np.mean(qa_em_scores)),
        "qa_consistency_f1": float(np.mean(qa_f1_scores)),
        "answer_recovery_f1": float(np.mean(answer_rec_scores))
    }

def main():
    print("="*80)
    print("STAGE 5: COMPARATIVE DOWNSTREAM QG & QA EVALUATION (MATCHING UPSTREAM SUITE)")
    print("="*80)

    pred_path = "lgsm_predictions.pkl"
    if not os.path.exists(pred_path):
        print(f"Error: Predictions file '{pred_path}' not found. Please run Stage 3 experiments first.")
        return

    print(f"Loading LGSM predictions from '{pred_path}'...")
    with open(pred_path, "rb") as f:
        analysis_data = pickle.load(f)
        
    val_records = analysis_data["val_records"]
    y_true = analysis_data["y_val_true"]
    probs_lgsm = analysis_data["probas"]
    
    print("\nRunning Zero-shot LLM Judge on validation split...")
    llm_classifier = LLMJudgeClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
    probs_llm = llm_classifier.predict_proba(val_records)
    
    df = pd.DataFrame({
        "question_id": [r["question_id"] for r in val_records],
        "question": [r.get("question", "") for r in val_records],
        "context": [r["context"] for r in val_records],
        "sentence_text": [r["sentence_text"] for r in val_records],
        "sentence_idx": [r["sentence_idx"] for r in val_records],
        "label": y_true,
        "prob_lgsm": probs_lgsm,
        "prob_llm": probs_llm
    })
    
    pipeline = DiscourseQGPipeline(device="cuda" if torch.cuda.is_available() else "cpu")
    rouge_calc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    grouped = df.groupby("question_id")
    print(f"Loaded predictions for {len(grouped)} unique contexts.")
    
    contexts = []
    gold_ref_questions = []
    
    oracle_sents = []
    baseline_sents = []
    lgsm_sents = []
    llm_sents = []
    
    q_ids = []
    lgsm_top1_correct = []
    llm_top1_correct = []
    
    for q_id, group in grouped:
        context = group["context"].iloc[0]
        ref_q = group["question"].iloc[0]
        
        # 1. Oracle Target Sentence (Ground truth label = 1)
        gt_rows = group[group["label"] == 1]
        if gt_rows.empty:
            continue
        oracle_sent = gt_rows.sort_values(by="prob_lgsm", ascending=False).iloc[0]["sentence_text"]
        
        # 2. Baseline Sentence (Random or first non-salient sentence)
        non_gt_rows = group[group["label"] == 0]
        baseline_sent = non_gt_rows.iloc[0]["sentence_text"] if not non_gt_rows.empty else oracle_sent
        
        # 3. LGSM Predicted top-1 sentence
        pred_lgsm_row = group.sort_values(by="prob_lgsm", ascending=False).iloc[0]
        lgsm_sent = pred_lgsm_row["sentence_text"]
        is_lgsm_correct = int(pred_lgsm_row["label"] == 1)
        
        # 4. LLM Judge Predicted top-1 sentence
        pred_llm_row = group.sort_values(by="prob_llm", ascending=False).iloc[0]
        llm_sent = pred_llm_row["sentence_text"]
        is_llm_correct = int(pred_llm_row["label"] == 1)
        
        contexts.append(context)
        gold_ref_questions.append(ref_q)
        oracle_sents.append(oracle_sent)
        baseline_sents.append(baseline_sent)
        lgsm_sents.append(lgsm_sent)
        llm_sents.append(llm_sent)
        
        q_ids.append(q_id)
        lgsm_top1_correct.append(is_lgsm_correct)
        llm_top1_correct.append(is_llm_correct)
        
    print(f"\n[{time.strftime('%X')}] Generating Questions across 4 Strategies (Batch Size = 16)...")
    oracle_qs = pipeline.generate_questions_batch(contexts, oracle_sents, batch_size=16)
    baseline_qs = pipeline.generate_questions_batch(contexts, baseline_sents, batch_size=16)
    lgsm_qs = pipeline.generate_questions_batch(contexts, lgsm_sents, batch_size=16)
    llm_qs = pipeline.generate_questions_batch(contexts, llm_sents, batch_size=16)
    
    print(f"[{time.strftime('%X')}] Answering Generated Questions (Batch Size = 16)...")
    oracle_ans = pipeline.answer_questions_batch(contexts, oracle_qs, batch_size=16)
    baseline_ans = pipeline.answer_questions_batch(contexts, baseline_qs, batch_size=16)
    lgsm_ans = pipeline.answer_questions_batch(contexts, lgsm_qs, batch_size=16)
    llm_ans = pipeline.answer_questions_batch(contexts, llm_qs, batch_size=16)
    
    print(f"\n[{time.strftime('%X')}] Computing Full Evaluation Metric Suite...")
    print("--> Evaluating Baseline QG...")
    m_baseline = evaluate_question_metrics(baseline_qs, gold_ref_questions, baseline_sents, baseline_ans, rouge_calc)
    
    print("--> Evaluating Zero-Shot LLM Judge QG...")
    m_llm = evaluate_question_metrics(llm_qs, gold_ref_questions, llm_sents, llm_ans, rouge_calc)
    
    print("--> Evaluating LGSM + DSNB Proposed Salience QG...")
    m_lgsm = evaluate_question_metrics(lgsm_qs, gold_ref_questions, lgsm_sents, lgsm_ans, rouge_calc)
    
    print("--> Evaluating Oracle Ground-Truth QG...")
    m_oracle = evaluate_question_metrics(oracle_qs, gold_ref_questions, oracle_sents, oracle_ans, rouge_calc)
    
    total = len(contexts)
    lgsm_top1_acc = np.mean(lgsm_top1_correct) if total > 0 else 0.0
    llm_top1_acc = np.mean(llm_top1_correct) if total > 0 else 0.0
    
    print("\n" + "="*80)
    print("STAGE 5 COMPLETE COMPARATIVE QG & QA EVALUATION METRICS TABLE")
    print("="*80)
    print(f"{'Strategy':<30} | {'ROUGE-L':<8} | {'BERTScore':<9} | {'QA-EM (%)':<9} | {'QA-F1 (%)':<9} | {'Ans-Rec (%)':<10}")
    print("-" * 90)
    print(f"{'Baseline (Non-Salient)':<30} | {m_baseline['rouge_l']:<8.4f} | {m_baseline['bert_score_f1']:<9.4f} | {m_baseline['qa_em']*100:<9.2f} | {m_baseline['qa_consistency_f1']*100:<9.2f} | {m_baseline['answer_recovery_f1']*100:<10.2f}")
    print(f"{'LLM Judge (Zero-shot)':<30} | {m_llm['rouge_l']:<8.4f} | {m_llm['bert_score_f1']:<9.4f} | {m_llm['qa_em']*100:<9.2f} | {m_llm['qa_consistency_f1']*100:<9.2f} | {m_llm['answer_recovery_f1']*100:<10.2f}")
    print(f"{'LGSM + DSNB (Proposed)':<30} | {m_lgsm['rouge_l']:<8.4f} | {m_lgsm['bert_score_f1']:<9.4f} | {m_lgsm['qa_em']*100:<9.2f} | {m_lgsm['qa_consistency_f1']*100:<9.2f} | {m_lgsm['answer_recovery_f1']*100:<10.2f}")
    print(f"{'Oracle (Ground-Truth Target)':<30} | {m_oracle['rouge_l']:<8.4f} | {m_oracle['bert_score_f1']:<9.4f} | {m_oracle['qa_em']*100:<9.2f} | {m_oracle['qa_consistency_f1']*100:<9.2f} | {m_oracle['answer_recovery_f1']*100:<10.2f}")
    print("="*80)
    
    # Save Detailed Markdown and CSV reports
    os.makedirs("docs", exist_ok=True)
    summary_df = pd.DataFrame([
        {"Strategy": "Baseline (Non-Salient)", **m_baseline},
        {"Strategy": "LLM Judge (Zero-shot)", **m_llm},
        {"Strategy": "LGSM + DSNB (Proposed)", **m_lgsm},
        {"Strategy": "Oracle (Ground-Truth Target)", **m_oracle}
    ])
    summary_df.to_csv("docs/qg_evaluation_summary_metrics.csv", index=False)
    
    with open("docs/qg_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("# Downstream QG & QA Agent Comparative Verification Report (Stage 5)\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("### Comprehensive Evaluation Metrics Suite\n")
        f.write("| Strategy | ROUGE-L | BERTScore F1 | QA Exact Match (%) | QA Consistency F1 (%) | Answer Recovery F1 (%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Baseline (Non-Salient)** | {m_baseline['rouge_l']:.4f} | {m_baseline['bert_score_f1']:.4f} | {m_baseline['qa_em']*100:.2f}% | {m_baseline['qa_consistency_f1']*100:.2f}% | {m_baseline['answer_recovery_f1']*100:.2f}% |\n")
        f.write(f"| **LLM Judge (Zero-shot)** | {m_llm['rouge_l']:.4f} | {m_llm['bert_score_f1']:.4f} | {m_llm['qa_em']*100:.2f}% | {m_llm['qa_consistency_f1']*100:.2f}% | {m_llm['answer_recovery_f1']*100:.2f}% |\n")
        f.write(f"| **LGSM + DSNB (Proposed)** | **{m_lgsm['rouge_l']:.4f}** | **{m_lgsm['bert_score_f1']:.4f}** | **{m_lgsm['qa_em']*100:.2f}%** | **{m_lgsm['qa_consistency_f1']*100:.2f}%** | **{m_lgsm['answer_recovery_f1']*100:.2f}%** |\n")
        f.write(f"| **Oracle (Ground Truth)** | {m_oracle['rouge_l']:.4f} | {m_oracle['bert_score_f1']:.4f} | {m_oracle['qa_em']*100:.2f}% | {m_oracle['qa_consistency_f1']*100:.2f}% | {m_oracle['answer_recovery_f1']*100:.2f}% |\n\n")
        
        f.write("### Top-1 Sentence Saliency Accuracy\n")
        f.write(f"- **LGSM Top-1 Saliency Accuracy**: `{lgsm_top1_acc*100:.2f}%`\n")
        f.write(f"- **LLM Judge Top-1 Saliency Accuracy**: `{llm_top1_acc*100:.2f}%`\n")

    print("\nSaved QG detailed report to docs/qg_evaluation_report.md and docs/qg_evaluation_summary_metrics.csv.")
    print("="*80)

if __name__ == "__main__":
    main()

