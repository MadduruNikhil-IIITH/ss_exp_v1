import os
import time
import spacy
import torch
import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict

# Feature extractors
from src.surprisal_features import SurprisalCalculator
from src.rst_features import DiscourseParserWrapper, RSTFeatureExtractor
from src.concreteness_features import ConcretenessScoreCalculator
from src.linguistic_features import extract_linguistic_features
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Classifiers
from src.classifiers.rule_based_rst import RuleBasedRSTClassifier
from src.classifiers.logistic_reg import TabularClassifierWrapper
from src.classifiers.hybrid_bert import HybridBERTClassifier
from src.classifiers.lgsm import LGSMSaliencyClassifier
from src.classifiers.llm_judge import LLMJudgeClassifier
from src.qg_pipeline import DiscourseQGPipeline

# Download VADER lexicon if missing
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize FastAPI App
app = FastAPI(title="SQuAD Sentence Salience QA System", description="Interactive QA Agent Interface")

# Mounting static files and templates
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for models (CPU Inference mode)
device = "cpu"
nlp = None
surprisal_calc = None
discourse_parser = None
rst_extractor = None
concrete_calc = None
sentiment_analyzer = None
ling_extractor = None

# Classifiers Cache
models = {}
qg_pipeline = None

class ContextRequest(BaseModel):
    context: str
    model_name: str  # 'lgsm', 'bert_pairwise', 'llm_judge', 'lr_combined', 'rst_rule_based'

class QGRequest(BaseModel):
    context: str
    sentence: str

class QARequest(BaseModel):
    context: str
    question: str
    target_sentence: str

@app.on_event("startup")
def load_resources():
    global nlp, surprisal_calc, discourse_parser, rst_extractor, concrete_calc, sentiment_analyzer, ling_extractor, qg_pipeline
    print("="*80)
    print("STARTING QA AGENT WEB INTERFACE SERVICE...")
    print("="*80)
    
    # Load spaCy
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    # Initialize Stage 2 Feature Extractors
    print("Initializing Feature Extractors (GPT-2, BERT, RST)...")
    surprisal_calc = SurprisalCalculator(causal_model_name="gpt2", masked_model_name="bert-base-uncased", device=device)
    discourse_parser = DiscourseParserWrapper(device=device)
    rst_extractor = RSTFeatureExtractor()
    concrete_calc = ConcretenessScoreCalculator()
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    # Initialize QG Pipeline
    print("Initializing QG & QA Agent solver...")
    qg_pipeline = DiscourseQGPipeline(device=device)
    
    # Load Model Checkpoints
    checkpoint_dir = "checkpoints"
    
    # 1. Rule-Based RST
    models["rst_rule_based"] = RuleBasedRSTClassifier()
    
    # 2. LR Combined (DSNB)
    lr_path = os.path.join(checkpoint_dir, "lr_combined_DSNB.joblib")
    if os.path.exists(lr_path):
        lr_wrapper = TabularClassifierWrapper(feature_mode="combined", use_soft_targets=False)
        lr_wrapper.load(lr_path)
        models["lr_combined"] = lr_wrapper
        print("Loaded LR Combined checkpoint.")
        
    # 3. Gated BERT (Pairwise)
    bert_path = os.path.join(checkpoint_dir, "bert_gated_all_Pairwise.pt")
    if os.path.exists(bert_path):
        bert_classifier = HybridBERTClassifier(mode="gated_all", device=device)
        bert_classifier.load(bert_path, device=device)
        models["bert_pairwise"] = bert_classifier
        print("Loaded Gated BERT checkpoint.")
        
    # 4. LGSM
    lgsm_path = os.path.join(checkpoint_dir, "lgsm.pt")
    if os.path.exists(lgsm_path):
        lgsm_classifier = LGSMSaliencyClassifier(pretrained_name="bert-base-uncased", device=device)
        lgsm_classifier.load(lgsm_path, device=device)
        models["lgsm"] = lgsm_classifier
        print("Loaded LGSM checkpoint.")
        
    # 5. LLM Judge (Zero-shot)
    models["llm_judge"] = LLMJudgeClassifier(device=device)
    print("Loaded LLM Judge.")
    print("="*80)

def extract_features_on_the_fly(context: str, sentences: List[Dict]) -> List[Dict]:
    """
    Runs the Stage 2 feature extraction pipeline on the fly for custom contexts.
    """
    # A. Causal Surprisal and Masked PLL
    try:
        surp_feats = surprisal_calc.extract_surprisal_features(sentences, context)
    except Exception as e:
        print(f"Warning: surprisal extraction failed: {e}")
        surp_feats = [{} for _ in sentences]
        
    # B. RST Discourse
    try:
        rst_root = discourse_parser.parse(context, sentences)
        rst_feats = rst_extractor.extract_rst_features(rst_root, sentences)
    except Exception as e:
        print(f"Warning: RST extraction failed: {e}")
        rst_feats = [{} for _ in sentences]
        
    # C. Linguistic, Concreteness and Sentiment
    merged_features = []
    for idx, sent in enumerate(sentences):
        text = sent["text"]
        
        # Extract linguistic, concreteness, and sentiment features
        ling_feats = extract_linguistic_features(text)
        cf_feats = concrete_calc.extract_concreteness_features(text)
        ling_feats.update(cf_feats)
        
        s_scores = sentiment_analyzer.polarity_scores(text)
        ling_feats["sentiment_polarity_pos"] = float(s_scores["pos"])
        ling_feats["sentiment_polarity_neg"] = float(s_scores["neg"])
        ling_feats["sentiment_polarity_neu"] = float(s_scores["neu"])
        ling_feats["sentiment_polarity_compound"] = float(s_scores["compound"])
        
        # Position features
        pos_feats = {
            "sentence_idx": float(idx),
            "position_ratio": float(idx) / max(1.0, len(sentences) - 1),
            "distance_from_end": float(len(sentences) - 1 - idx),
        }
        
        feats = {}
        feats.update(pos_feats)
        feats.update(ling_feats)
        if idx < len(surp_feats):
            feats.update(surp_feats[idx])
        if idx < len(rst_feats):
            feats.update(rst_feats[idx])
            
        merged_features.append(feats)
        
    return merged_features

from fastapi.responses import HTMLResponse, FileResponse

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return FileResponse("templates/index.html")

@app.post("/predict_saliency")
def predict_saliency(req: ContextRequest):
    context = req.context.strip()
    model_name = req.model_name
    
    if not context:
        raise HTTPException(status_code=400, detail="Context cannot be empty.")
        
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not loaded.")
        
    # 1. Segment context into sentences
    doc = nlp(context)
    sentences_boundaries = []
    char_offset = 0
    for idx, sent in enumerate(doc.sents):
        text = sent.text
        start_char = context.find(text, char_offset)
        if start_char == -1:
            start_char = char_offset
        end_char = start_char + len(text)
        char_offset = end_char
        
        sentences_boundaries.append({
            "sentence_idx": idx,
            "text": text,
            "start_char": start_char,
            "end_char": end_char
        })
        
    if not sentences_boundaries:
         raise HTTPException(status_code=400, detail="No sentences detected in context.")
         
    # 2. Extract features (needed for all trained classifiers)
    print(f"Extracting features for {len(sentences_boundaries)} input sentences...")
    features_list = extract_features_on_the_fly(context, sentences_boundaries)
    
    # 3. Format records for wrapper predict functions
    records = []
    for i, sent in enumerate(sentences_boundaries):
        records.append({
            "sentence_text": sent["text"],
            "sentence_idx": i,
            "question_id": "web_interface_ctx",
            "context": context,
            "binary_label": 0,
            "features": features_list[i]
        })
        
    # 4. Predict target model & Zero-shot LLM Judge for comparison
    classifier = models[model_name]
    
    if model_name == "rst_rule_based":
        rst_feats = [r["features"] for r in records]
        probs = classifier.predict_proba(rst_feats).tolist()
    elif model_name == "lr_combined":
        probs = classifier.predict_proba(records).tolist()
    elif model_name == "bert_pairwise":
        probs = classifier.predict_proba(records, batch_size=32).tolist()
    elif model_name == "lgsm":
        probs, _ = classifier.predict_proba(records, batch_size=4)
        probs = probs.tolist()
    elif model_name == "llm_judge":
        probs = classifier.predict_proba(records).tolist()
        
    # Also get Zero-shot LLM Judge probabilities for side-by-side comparison
    if model_name != "llm_judge" and "llm_judge" in models:
        try:
            llm_judge_probs = models["llm_judge"].predict_proba(records).tolist()
        except Exception:
            llm_judge_probs = [0.35] * len(records)
    else:
        llm_judge_probs = probs

    # Zip output
    saliency_outputs = []
    for i, sent in enumerate(sentences_boundaries):
        saliency_outputs.append({
            "sentence_idx": i,
            "text": sent["text"],
            "probability": float(probs[i]),
            "llm_judge_probability": float(llm_judge_probs[i]),
            "features": {
                "Surprisal Deletion Drop": f"{features_list[i].get('surp_deletion_drop', 0.0):.4f}",
                "RST Mean Depth": f"{features_list[i].get('rst_mean_depth', 0.0):.2f}",
                "Syntax Depth": f"{features_list[i].get('max_parse_depth', 0.0):.2f}",
                "Readability index": f"{features_list[i].get('flesch_reading_ease', 0.0):.2f}"
            }
        })
        
    return {
        "sentences": saliency_outputs,
        "selected_idx": int(np.argmax(probs))
    }

@app.post("/generate_question")
def generate_question(req: QGRequest):
    context = req.context.strip()
    sentence = req.sentence.strip()
    
    if not context or not sentence:
        raise HTTPException(status_code=400, detail="Context and target sentence cannot be empty.")
        
    try:
        question = qg_pipeline.generate_question(context, sentence)
        return {"question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answer_question")
def answer_question(req: QARequest):
    context = req.context.strip()
    question = req.question.strip()
    target_sentence = req.target_sentence.strip()
    
    if not context or not question or not target_sentence:
        raise HTTPException(status_code=400, detail="All inputs are required.")
        
    try:
        predicted_answer = qg_pipeline.answer_question(context, question)
        verified = qg_pipeline.verify_question(target_sentence, predicted_answer)
        return {
            "predicted_answer": predicted_answer,
            "verified": bool(verified)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_quiz")
def generate_quiz(req: ContextRequest):
    context = req.context.strip()
    model_name = req.model_name
    
    if not context:
        raise HTTPException(status_code=400, detail="Context cannot be empty.")
        
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not loaded.")
        
    # 1. Segment context into sentences
    doc = nlp(context)
    sentences_boundaries = []
    char_offset = 0
    for idx, sent in enumerate(doc.sents):
        text = sent.text
        start_char = context.find(text, char_offset)
        if start_char == -1:
            start_char = char_offset
        end_char = start_char + len(text)
        char_offset = end_char
        
        sentences_boundaries.append({
            "sentence_idx": idx,
            "text": text,
            "start_char": start_char,
            "end_char": end_char
        })
        
    if not sentences_boundaries:
         raise HTTPException(status_code=400, detail="No sentences detected in context.")
         
    # 2. Extract features on the fly
    features_list = extract_features_on_the_fly(context, sentences_boundaries)
    
    # 3. Format records
    records = []
    for i, sent in enumerate(sentences_boundaries):
        records.append({
            "sentence_text": sent["text"],
            "sentence_idx": i,
            "question_id": "quiz_generation_ctx",
            "context": context,
            "binary_label": 0,
            "features": features_list[i]
        })
        
    # 4. Predict probabilities
    classifier = models[model_name]
    if model_name == "rst_rule_based":
        rst_feats = [r["features"] for r in records]
        probs = classifier.predict_proba(rst_feats).tolist()
    elif model_name == "lr_combined":
        probs = classifier.predict_proba(records).tolist()
    elif model_name == "bert_pairwise":
        probs = classifier.predict_proba(records, batch_size=32).tolist()
    elif model_name == "lgsm":
        probs, _ = classifier.predict_proba(records, batch_size=4)
        probs = probs.tolist()
    elif model_name == "llm_judge":
        probs = classifier.predict_proba(records).tolist()
        
    # 5. Filter top salient sentences (Up to Top 5 for long passages, Top 2-3 for short passages)
    if len(sentences_boundaries) >= 10:
        max_k = 5
    else:
        max_k = 3
    top_k_count = min(max_k, max(2, len([p for p in probs if p >= 0.30])))
    top_k_indices = list(np.argsort(probs)[::-1][:top_k_count])
    salient_indices = sorted(top_k_indices)
        
    salient_sents = [sentences_boundaries[idx]["text"] for idx in salient_indices]
    salient_contexts = [context] * len(salient_sents)
    
    # 6. Batched QG on GPU
    generated_qs = qg_pipeline.generate_questions_batch(salient_contexts, salient_sents, batch_size=16)
    
    # 7. Batched QA Solver verification
    predicted_answers = qg_pipeline.answer_questions_batch(salient_contexts, generated_qs, batch_size=16)
    
    # 8. Verify
    quiz_items = []
    for i, idx in enumerate(salient_indices):
        original_sentence = salient_sents[i]
        question = generated_qs[i]
        predicted_answer = predicted_answers[i]
        verified = qg_pipeline.verify_question(original_sentence, predicted_answer)
        
        quiz_items.append({
            "sentence_idx": idx,
            "target_sentence": original_sentence,
            "question": question,
            "predicted_answer": predicted_answer,
            "verified": bool(verified)
        })
        
    return {
        "quiz_items": quiz_items
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
