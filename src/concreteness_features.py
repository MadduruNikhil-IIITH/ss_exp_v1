import os
import urllib.request
import pandas as pd
import numpy as np
import spacy

# Load spaCy model for lemmatization
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    nlp = spacy.blank("en")

LEXICON_URL = "https://raw.githubusercontent.com/ArtsEngine/concreteness/refs/heads/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
LEXICON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Concreteness_ratings_Brysbaert_et_al_BRM.txt")

class ConcretenessScoreCalculator:
    def __init__(self):
        self.concreteness_dict = {}
        self._load_lexicon()

    def _load_lexicon(self):
        """
        Loads the Brysbaert concreteness dictionary, downloading it if not present locally.
        """
        if not os.path.exists(LEXICON_PATH):
            print(f"Brysbaert concreteness dictionary not found locally. Downloading from {LEXICON_URL}...")
            try:
                urllib.request.urlretrieve(LEXICON_URL, LEXICON_PATH)
                print("Download completed successfully.")
            except Exception as e:
                print(f"Error downloading Brysbaert concreteness dictionary: {e}")
                return

        if os.path.exists(LEXICON_PATH):
            try:
                # The file is tab-separated
                df = pd.read_csv(LEXICON_PATH, sep="\t")
                # Map lowercase word to Conc.M (Mean Concreteness rating)
                self.concreteness_dict = dict(zip(df["Word"].astype(str).str.lower(), df["Conc.M"].astype(float)))
                print(f"Loaded {len(self.concreteness_dict)} words from Brysbaert concreteness norms.")
            except Exception as e:
                print(f"Error loading Brysbaert concreteness dictionary: {e}")

    def extract_concreteness_features(self, text):
        """
        Calculates the mean, max, min, and standard deviation of concreteness scores 
        for non-stopwords in the text. Defaults to 3.0 (neutral value in 1-5 scale) if no words found.
        """
        doc = nlp(text)
        scores = []
        for token in doc:
            if not token.is_punct and not token.is_space and not token.is_stop:
                word_lower = token.text.lower()
                lemma_lower = token.lemma_.lower()
                
                # Check both the word and its lemma
                if word_lower in self.concreteness_dict:
                    scores.append(self.concreteness_dict[word_lower])
                elif lemma_lower in self.concreteness_dict:
                    scores.append(self.concreteness_dict[lemma_lower])
                    
        if not scores:
            return {
                "concrete_mean": 3.0,
                "concrete_max": 3.0,
                "concrete_min": 3.0,
                "concrete_std": 0.0
            }
            
        return {
            "concrete_mean": float(np.mean(scores)),
            "concrete_max": float(np.max(scores)),
            "concrete_min": float(np.min(scores)),
            "concrete_std": float(np.std(scores)) if len(scores) > 1 else 0.0
        }
