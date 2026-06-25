import spacy
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load spaCy model (we disable parser but keep tagger and ner for Named Entity matching)
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
except OSError:
    nlp = spacy.blank("en")

class AlignmentFeatureExtractor:
    def __init__(self, sbert_model_name="all-MiniLM-L6-v2", device="cuda"):
        sbert_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        print(f"Loading SBERT model '{sbert_model_name}' on {sbert_device}...")
        self.sbert = SentenceTransformer(sbert_model_name, device=sbert_device)
        
    def get_expected_entities(self, question):
        """
        Maps standard question keywords to expected named entity types from spaCy.
        """
        q_lower = question.lower()
        expected = []
        if "who" in q_lower or "whom" in q_lower or "whose" in q_lower:
            expected = ["PERSON", "ORG"]
        elif "when" in q_lower or "date" in q_lower or "year" in q_lower:
            expected = ["DATE", "TIME"]
        elif "where" in q_lower or "place" in q_lower or "city" in q_lower or "country" in q_lower or "state" in q_lower:
            expected = ["GPE", "LOC", "FAC"]
        elif "how many" in q_lower or "how much" in q_lower or "number" in q_lower or "amount" in q_lower:
            expected = ["CARDINAL", "QUANTITY", "MONEY", "PERCENT"]
        return expected

    def extract_alignment_features(self, question, sentence_boundaries):
        """
        Extracts lexical overlap, semantic similarity, and Named Entity match features
        for a list of sentences relative to the question.
        """
        if not sentence_boundaries:
            return []
            
        q_doc = nlp(question)
        # Filter lemmas for lexical overlap (exclude punctuation, spaces, and stopwords)
        q_lemmas = set(token.lemma_.lower() for token in q_doc if not token.is_punct and not token.is_space and not token.is_stop)
        
        expected_ents = self.get_expected_entities(question)
        
        # SBERT Embeddings for semantic similarity
        q_emb = self.sbert.encode([question], convert_to_numpy=True)
        sent_texts = [sent["text"] for sent in sentence_boundaries]
        sent_embs = self.sbert.encode(sent_texts, convert_to_numpy=True)
        
        # Cosine similarity
        similarities = cosine_similarity(sent_embs, q_emb).flatten()
        
        alignment_features = []
        for idx, sent in enumerate(sentence_boundaries):
            sent_text = sent["text"]
            s_doc = nlp(sent_text)
            s_lemmas = set(token.lemma_.lower() for token in s_doc if not token.is_punct and not token.is_space and not token.is_stop)
            
            # 1. Lexical Overlap (Jaccard and absolute count)
            intersection = q_lemmas.intersection(s_lemmas)
            union = q_lemmas.union(s_lemmas)
            jaccard = len(intersection) / max(1, len(union))
            match_count = float(len(intersection))
            
            # Approximate ROUGE-L using Longest Common Subsequence (LCS)
            q_list = [token.lemma_.lower() for token in q_doc if not token.is_punct and not token.is_space]
            s_list = [token.lemma_.lower() for token in s_doc if not token.is_punct and not token.is_space]
            lcs_len = self.compute_lcs_length(q_list, s_list)
            rouge_l_recall = lcs_len / max(1, len(q_list))
            
            # 2. Named Entity Match
            ne_match = 0
            if expected_ents:
                sent_ents = [ent.label_ for ent in s_doc.ents]
                if any(ent in expected_ents for ent in sent_ents):
                    ne_match = 1
            
            # 3. Semantic similarity
            sem_sim = float(similarities[idx])
            
            alignment_features.append({
                "align_jaccard": jaccard,
                "align_match_count": match_count,
                "align_rouge_l_recall": rouge_l_recall,
                "align_ne_match": float(ne_match),
                "align_sem_sim": sem_sim
            })
            
        return alignment_features

    def compute_lcs_length(self, x, y):
        """
        Helper to compute the length of the Longest Common Subsequence.
        """
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
