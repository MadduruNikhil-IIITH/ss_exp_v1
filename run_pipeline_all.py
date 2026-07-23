import os
import subprocess
import sys
import time

def run_cmd(args):
    print("\n" + "="*80)
    print(f"RUNNING PIPELINE STAGE: {' '.join(args)}")
    print("="*80)
    start_time = time.time()
    
    # Run using the active conda python interpreter
    result = subprocess.run([sys.executable] + args, check=True)
    duration = time.time() - start_time
    print(f"STAGE COMPLETED SUCCESSFULLY IN {duration/60:.2f} MINUTES.")
    return duration

def main():
    print("="*80)
    print("MASTER EXECUTION PIPELINE: RUNNING ALL STAGES (1000 CONTEXTS, 80/20 SPLIT)")
    print("="*80)
    
    total_start = time.time()
    
    # Stage 1: Dataset Prep (1000 contexts, 80/20 split, 20% random LLM audit) - ALREADY CACHED
    # run_cmd(["run_stage1_dataset_prep.py", "--total_contexts", "1000", "--num_train_contexts", "800", "--num_val_contexts", "200", "--audit_ratio", "0.20"])
    
    # Stage 2: Feature Extraction (GPT-2, BERT, ISANLP RST) - ALREADY CACHED
    # run_cmd(["run_stage2_feature_extraction.py"])
    
    # Stage 3: Train Classifiers (LGSM, Gated BERT, Tabular LR)
    run_cmd(["run_stage3_experiments.py"])
    
    # Stage 4: Cross-Validation & Diagnostics
    run_cmd(["run_stage4_cross_validation.py"])
    run_cmd(["run_stage4_diagnostics.py"])
    run_cmd(["run_stage4_evaluation.py"])
    
    # Stage 5: Downstream QG Evaluation (Oracle vs LGSM vs LLM Judge)
    run_cmd(["run_stage5_qg_eval.py"])
    
    # Run Interpretability Studies & Statistical Diagnostics
    run_cmd(["analysis/run_pca_analysis.py"])
    run_cmd(["analysis/run_post_lasso.py"])
    run_cmd(["analysis/run_gating_analysis.py"])
    run_cmd(["analysis/run_uid_analysis.py"])
    
    total_duration = time.time() - total_start
    print("\n" + "="*80)
    print(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_duration/60:.2f} MINUTES!")
    print("="*80)

if __name__ == "__main__":
    main()
