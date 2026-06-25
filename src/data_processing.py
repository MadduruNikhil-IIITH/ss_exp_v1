import numpy as np
import pandas as pd
import spacy
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from tqdm import tqdm

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    # We enable sentencizer for sentence boundary detection
    nlp.add_pipe("sentencizer")
except OSError:
    # If not installed, we will load a basic sentencizer
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

def get_sentence_boundaries(context):
    """
    Splits context into sentences and records text and (start_char, end_char) boundaries.
    """
    doc = nlp(context)
    sentences = []
    for i, sent in enumerate(doc.sents):
        # We strip whitespaces but keep character offsets relative to original context
        start = sent.start_char
        end = sent.end_char
        text = sent.text.strip()
        sentences.append({
            "sentence_idx": i,
            "text": text,
            "start_char": start,
            "end_char": end
        })
    return sentences

def compute_tfidf_similarity(question, sentences):
    """
    Computes cosine similarity between the TF-IDF vector of the question and each sentence.
    """
    if not sentences:
        return []
    
    texts = [sent["text"] for sent in sentences] + [question]
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        # Slices: last row is the question, rest are sentences
        q_vec = tfidf_matrix[-1]
        sent_vecs = tfidf_matrix[:-1]
        sims = cosine_similarity(sent_vecs, q_vec).flatten()
        return sims.tolist()
    except Exception:
        # Fallback if TF-IDF fails (e.g. empty vocabulary)
        return [0.0] * len(sentences)

def has_token_overlap(ans_text, sent_text):
    """
    Checks if there is at least one non-stopword, alphanumeric token overlap 
    between the answer text and the sentence to filter out boundary-spilling noise.
    """
    ans_doc = nlp(ans_text.lower())
    sent_doc = nlp(sent_text.lower())
    
    ans_tokens = {t.text for t in ans_doc if not t.is_stop and t.is_alpha}
    sent_tokens = {t.text for t in sent_doc if not t.is_stop and t.is_alpha}
    
    # If the answer is purely stopwords or non-alphanumeric, fallback to exact string matching
    if not ans_tokens:
        return ans_text.lower() in sent_text.lower()
        
    return bool(ans_tokens.intersection(sent_tokens))

def build_silver_squad_dataset(num_contexts=100, split="train"):
    """
    Downloads SQuAD v1.1, processes contexts, matches answer spans to sentence boundaries,
    and returns a DataFrame containing QA pairs with sentence level labels and metadata.
    """
    print(f"Loading SQuAD v1.1 ({split} split)...")
    dataset = load_dataset("rajpurkar/squad", split=split)
    
    # Group by context to de-duplicate parsing
    contexts_map = {}
    for item in dataset:
        context = item["context"]
        if context not in contexts_map:
            contexts_map[context] = []
        contexts_map[context].append(item)
    
    # Process only a subset of contexts for performance
    unique_contexts = list(contexts_map.keys())[:num_contexts]
    print(f"Processing {len(unique_contexts)} unique contexts ({len(dataset)} total QA pairs in SQuAD)...")
    
    processed_records = []
    
    for context in tqdm(unique_contexts, desc="Silver labeling contexts"):
        sentences = get_sentence_boundaries(context)
        if not sentences:
            continue
            
        qa_pairs = contexts_map[context]
        for qa in qa_pairs:
            q_id = qa["id"]
            question = qa["question"]
            answers = qa["answers"]
            
            # 1. Exact-Index Binary Labels with Token-Level Intersection Filter
            salient_indices = set()
            for ans_text, ans_start in zip(answers["text"], answers["answer_start"]):
                ans_end = ans_start + len(ans_text)
                for sent in sentences:
                    # Check for non-empty character boundary intersection
                    if max(sent["start_char"], ans_start) < min(sent["end_char"], ans_end):
                        # Filter out punctuation and boundary noise via token overlap check
                        if has_token_overlap(ans_text, sent["text"]):
                            salient_indices.add(sent["sentence_idx"])
            
            # Primary answer sentence (first annotated answer)
            primary_ans_idx = 0
            if salient_indices:
                primary_ans_idx = min(salient_indices)
            
            # Compute TF-IDF question-sentence similarities
            tfidf_sims = compute_tfidf_similarity(question, sentences)
            
            # Process each sentence in the context for this question
            for idx, sent in enumerate(sentences):
                # Binary salience
                is_salient = 1 if idx in salient_indices else 0
                
                # Distance-based decay label
                dist = min([abs(idx - s_idx) for s_idx in salient_indices]) if salient_indices else abs(idx - primary_ans_idx)
                soft_decay = 0.5 ** dist
                
                # Hybrid semantic label (decay + TF-IDF similarity)
                sim_score = tfidf_sims[idx] if idx < len(tfidf_sims) else 0.0
                soft_hybrid = 0.7 * soft_decay + 0.3 * sim_score
                
                processed_records.append({
                    "question_id": q_id,
                    "question": question,
                    "context": context,
                    "sentence_idx": idx,
                    "sentence_text": sent["text"],
                    "start_char": sent["start_char"],
                    "end_char": sent["end_char"],
                    "binary_label": is_salient,
                    "soft_label_decay": soft_decay,
                    "soft_label_hybrid": soft_hybrid
                })
                
    return pd.DataFrame(processed_records)

def apply_pairwise_balancing(input_data):
    """
    Transforms the SQuAD sentence salience dataset into balanced sentence pairs.
    Each row in the returned list/DataFrame contains:
    - sentence_1, sentence_2
    - features_1, features_2
    - label (1 if sentence_1 is salient, 0 if sentence_2 is salient)
    This is applied only to the training set.
    """
    is_list = isinstance(input_data, list)
    df = pd.DataFrame(input_data) if is_list else input_data
    
    print("Applying pairwise balancing on training data...")
    pairwise_records = []
    
    # Group by question_id to keep context pairs aligned
    grouped = df.groupby("question_id")
    for q_id, group in tqdm(grouped, desc="Applying pairwise balancing"):
        # Separate salient and non-salient sentences
        salient_rows = group[group["binary_label"] == 1]
        non_salient_rows = group[group["binary_label"] == 0]
        
        if salient_rows.empty or non_salient_rows.empty:
            continue
            
        for _, s_row in salient_rows.iterrows():
            for _, ns_row in non_salient_rows.iterrows():
                # Randomly decide order to keep labels balanced (50% 1, 50% 0)
                if np.random.rand() > 0.5:
                    pairwise_records.append({
                        "question_id": q_id,
                        "question": s_row["question"],
                        "context": s_row["context"],
                        "s1_text": s_row["sentence_text"],
                        "s2_text": ns_row["sentence_text"],
                        "s1_features": s_row.get("features", None),
                        "s2_features": ns_row.get("features", None),
                        "s1_idx": s_row["sentence_idx"],
                        "s2_idx": ns_row["sentence_idx"],
                        "label": 1  # s1 is salient
                    })
                else:
                    pairwise_records.append({
                        "question_id": q_id,
                        "question": s_row["question"],
                        "context": s_row["context"],
                        "s1_text": ns_row["sentence_text"],
                        "s2_text": s_row["sentence_text"],
                        "s1_features": ns_row.get("features", None),
                        "s2_features": s_row.get("features", None),
                        "s1_idx": ns_row["sentence_idx"],
                        "s2_idx": s_row["sentence_idx"],
                        "label": 0  # s2 is salient
                    })
                    
    df_res = pd.DataFrame(pairwise_records)
    return df_res.to_dict(orient="records") if is_list else df_res

def apply_cluster_balancing(input_data, feature_cols=None):
    """
    Applies cluster-based undersampling to balance salient vs. non-salient classes.
    Clusters the majority (non-salient) sentences using K-Means and samples representatives.
    If features are not present or feature_cols is None, it uses sentence lengths/positions.
    """
    is_list = isinstance(input_data, list)
    df = pd.DataFrame(input_data) if is_list else input_data
    
    print("Applying cluster-based balancing...")
    balanced_dfs = []
    
    # Perform balancing per question to keep contexts intact
    grouped = df.groupby("question_id")
    for q_id, group in tqdm(grouped, desc="Applying cluster balancing"):
        salient_rows = group[group["binary_label"] == 1]
        non_salient_rows = group[group["binary_label"] == 0]
        
        num_positives = len(salient_rows)
        num_negatives = len(non_salient_rows)
        
        if num_positives == 0:
            continue
        if num_negatives <= num_positives:
            balanced_dfs.append(group)
            continue
            
        # Select features for clustering
        if feature_cols is not None:
            # Check if features are stored in a dict or columns
            if isinstance(group.iloc[0].get("features"), dict):
                X = np.array([[row["features"].get(c, 0.0) for c in feature_cols] for _, row in non_salient_rows.iterrows()])
            else:
                X = non_salient_rows[feature_cols].values
        else:
            # Fallback: cluster based on sentence index and text length
            X = np.array([[row["sentence_idx"], len(row["sentence_text"])] for _, row in non_salient_rows.iterrows()])
            
        # Standardize X briefly to help K-Means
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8
        X_scaled = (X - X_mean) / X_std
        
        n_clusters = min(num_positives, len(non_salient_rows))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        # Find closest representative for each cluster
        selected_indices = []
        for i in range(n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) == 0:
                continue
            # Distance from center
            distances = np.linalg.norm(X_scaled[cluster_indices] - kmeans.cluster_centers_[i], axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(non_salient_rows.index[closest_idx])
            
        selected_negatives = non_salient_rows.loc[selected_indices]
        balanced_dfs.append(pd.concat([salient_rows, selected_negatives]))
        
    if not balanced_dfs:
        df_res = pd.DataFrame(columns=df.columns)
    else:
        df_res = pd.concat(balanced_dfs).reset_index(drop=True)
        
    return df_res.to_dict(orient="records") if is_list else df_res

def apply_rst_balancing(input_data):
    """
    Applies RST-Neighborhood balancing to select representative hard negatives.
    For each positive sentence in a question context, we select negative sentences
    that are closest in the discourse tree (rhetorical depth) and linear passage index.
    """
    is_list = isinstance(input_data, list)
    df = pd.DataFrame(input_data) if is_list else input_data
    
    print("Applying RST-Neighborhood balancing...")
    balanced_dfs = []
    
    # Perform balancing per question to keep contexts intact
    grouped = df.groupby("question_id")
    for q_id, group in tqdm(grouped, desc="Applying RST-Neighborhood balancing"):
        salient_rows = group[group["binary_label"] == 1]
        non_salient_rows = group[group["binary_label"] == 0]
        
        num_positives = len(salient_rows)
        num_negatives = len(non_salient_rows)
        
        if num_positives == 0:
            continue
        if num_negatives <= num_positives:
            balanced_dfs.append(group)
            continue
            
        # For each negative sentence, calculate its distance to the closest positive sentence
        selected_indices = []
        neg_candidates = []
        
        for idx_row, ns_row in non_salient_rows.iterrows():
            ns_idx = ns_row["sentence_idx"]
            ns_feats = ns_row.get("features", {})
            ns_depth = ns_feats.get("rst_mean_depth", 0.0)
            
            # Find minimum distance to any positive sentence in this context
            min_dist = float('inf')
            for _, s_row in salient_rows.iterrows():
                s_idx = s_row["sentence_idx"]
                s_feats = s_row.get("features", {})
                s_depth = s_feats.get("rst_mean_depth", 0.0)
                
                # Combine linear index distance and hierarchy depth difference
                pos_dist = abs(ns_idx - s_idx)
                rst_dist = abs(ns_depth - s_depth)
                
                # Distance score
                dist = 0.5 * pos_dist + 0.5 * rst_dist
                if dist < min_dist:
                    min_dist = dist
                    
            neg_candidates.append((idx_row, min_dist))
            
        # Sort negative candidates by distance ascending (closest first)
        neg_candidates = sorted(neg_candidates, key=lambda x: x[1])
        
        # Select the top num_positives negatives
        selected_names = [item[0] for item in neg_candidates[:num_positives]]
        selected_negatives = non_salient_rows.loc[selected_names]
        
        balanced_dfs.append(pd.concat([salient_rows, selected_negatives]))
        
    if not balanced_dfs:
        df_res = pd.DataFrame(columns=df.columns)
    else:
        df_res = pd.concat(balanced_dfs).reset_index(drop=True)
        
    return df_res.to_dict(orient="records") if is_list else df_res

def apply_dsnb_balancing(input_data):
    """
    Applies Discourse-Semantic Neighborhood Balancing (DSNB) to select hard negatives.
    For each positive sentence in a question context, we select negative sentences
    that are positionally close, semantically aligned to the question, and have
    similar discourse tree depth.
    """
    is_list = isinstance(input_data, list)
    df = pd.DataFrame(input_data) if is_list else input_data
    
    print("Applying Discourse-Semantic Neighborhood balancing (DSNB)...")
    balanced_dfs = []
    
    # Perform balancing per question to keep contexts intact
    grouped = df.groupby("question_id")
    for q_id, group in tqdm(grouped, desc="Applying DSNB balancing"):
        salient_rows = group[group["binary_label"] == 1]
        non_salient_rows = group[group["binary_label"] == 0]
        
        num_positives = len(salient_rows)
        num_negatives = len(non_salient_rows)
        
        if num_positives == 0:
            continue
        if num_negatives <= num_positives:
            balanced_dfs.append(group)
            continue
            
        # Standardize negative features for max depth scaling
        all_depths = [row.get("features", {}).get("rst_mean_depth", 0.0) for _, row in group.iterrows()]
        max_depth = max(all_depths) if all_depths else 1.0
        max_depth = max(1.0, max_depth)
        
        neg_candidates = []
        
        for idx_row, ns_row in non_salient_rows.iterrows():
            ns_idx = ns_row["sentence_idx"]
            ns_feats = ns_row.get("features", {})
            ns_depth = ns_feats.get("rst_mean_depth", 0.0)
            
            # Semantic similarity to question (SBERT score)
            sem_sim = ns_feats.get("align_sem_sim", 0.0)
            # Clip between [0, 1] for normalization
            w_sem = max(0.0, min(1.0, sem_sim))
            
            # Find maximum hardness score relative to any positive sentence
            max_hardness = -1.0
            for _, s_row in salient_rows.iterrows():
                s_idx = s_row["sentence_idx"]
                s_feats = s_row.get("features", {})
                s_depth = s_feats.get("rst_mean_depth", 0.0)
                
                # 1. Positional Proximity: 0.5^distance
                dist_pos = abs(ns_idx - s_idx)
                w_pos = 0.5 ** dist_pos
                
                # 2. RST Discourse depth similarity
                dist_rst = abs(ns_depth - s_depth)
                w_rst = 1.0 - (dist_rst / max_depth)
                w_rst = max(0.0, w_rst)
                
                # Combined Hardness Score: 0.4*Pos + 0.4*Sem + 0.2*RST
                score = 0.4 * w_pos + 0.4 * w_sem + 0.2 * w_rst
                if score > max_hardness:
                    max_hardness = score
                    
            neg_candidates.append((idx_row, max_hardness))
            
        # Sort negative candidates by hardness score descending (highest/hardest first)
        neg_candidates = sorted(neg_candidates, key=lambda x: x[1], reverse=True)
        
        # Select the top num_positives negatives
        selected_names = [item[0] for item in neg_candidates[:num_positives]]
        selected_negatives = non_salient_rows.loc[selected_names]
        
        balanced_dfs.append(pd.concat([salient_rows, selected_negatives]))
        
    if not balanced_dfs:
        df_res = pd.DataFrame(columns=df.columns)
    else:
        df_res = pd.concat(balanced_dfs).reset_index(drop=True)
        
    return df_res.to_dict(orient="records") if is_list else df_res

def classify_feature_keys(feature_keys):
    """
    Groups feature keys into categories: rst, surprisal, alignment, and linguistic.
    """
    feature_keys = list(feature_keys)
    rst_cols = [k for k in feature_keys if k.startswith(("rst_", "rel_rst_", "psg_rst_"))]
    surp_cols = [k for k in feature_keys if k.startswith(("surp_", "rel_surp_", "psg_surp_"))]
    align_cols = [k for k in feature_keys if k.startswith("align_")]
    
    exclude_prefixes = ("rst_", "rel_rst_", "psg_rst_", "surp_", "rel_surp_", "psg_surp_", "align_")
    ling_cols = [k for k in feature_keys if not k.startswith(exclude_prefixes)]
    
    return {
        "rst": rst_cols,
        "surprisal": surp_cols,
        "alignment": align_cols,
        "linguistic": ling_cols,
        "all": feature_keys
    }
