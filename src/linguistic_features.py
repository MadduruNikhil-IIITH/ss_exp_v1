import spacy
import textstat
import re
from collections import Counter

# Load spaCy model for POS tagging, dependency parsing, and tokenization
# We keep the parser enabled for syntactic tree analysis but disable NER
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    # Fallback to blank model with sentencizer
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

# Compile regular expressions for discourse markers and pronouns
CAUSAL_MARKERS = re.compile(
    r'\b(because|since|as|therefore|thus|hence|consequently|so|'
    r'accordingly|due to|owing to|as a result|for this reason)\b', 
    re.IGNORECASE
)
CONTRAST_MARKERS = re.compile(
    r'\b(but|however|although|though|despite|nevertheless|nonetheless|'
    r'yet|still|whereas|while|on the other hand|in contrast|conversely)\b', 
    re.IGNORECASE
)
ADDITION_MARKERS = re.compile(
    r'\b(and|also|moreover|furthermore|additionally|besides|in addition|'
    r'as well as|not only|similarly|likewise)\b', 
    re.IGNORECASE
)

PRON_1ST = re.compile(r'\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b', re.IGNORECASE)
PRON_2ND = re.compile(r'\b(you|your|yours|yourself|yourselves)\b', re.IGNORECASE)
PRON_3RD = re.compile(r'\b(he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves)\b', re.IGNORECASE)

def extract_linguistic_features(sentence_text):
    """
    Extracts a set of lexical, morphosyntactic, syntactic, and readability features
    for a given sentence, including discourse markers and syntactic tree stats.
    """
    doc = nlp(sentence_text)
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    word_count = len(words)
    char_count = len(sentence_text)
    total_tokens = max(1, len(doc))
    
    # 1. Basic lexical stats
    avg_word_length = sum(len(w) for w in words) / max(1, word_count)
    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / max(1, word_count)
    
    stopwords = [token for token in doc if token.is_stop]
    stopword_ratio = len(stopwords) / max(1, len(doc))
    
    # 2. POS tag ratios
    pos_counts = Counter(token.pos_ for token in doc)
    pos_ratios = {
        "noun_ratio": (pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)) / total_tokens,
        "verb_ratio": pos_counts.get("VERB", 0) / total_tokens,
        "adj_ratio": pos_counts.get("ADJ", 0) / total_tokens,
        "adv_ratio": pos_counts.get("ADV", 0) / total_tokens,
        "pron_ratio": pos_counts.get("PRON", 0) / total_tokens,
        "prep_ratio": pos_counts.get("ADP", 0) / total_tokens,
        "conj_ratio": (pos_counts.get("CONJ", 0) + pos_counts.get("CCONJ", 0) + pos_counts.get("SCONJ", 0)) / total_tokens,
    }
    
    # 3. Casing ratios
    capital_chars = sum(1 for c in sentence_text if c.isupper())
    cap_ratio = capital_chars / max(1, char_count)
    
    title_words = sum(1 for w in words if w.istitle())
    title_ratio = title_words / max(1, word_count)
    
    # 4. Readability scores (via textstat)
    try:
        flesch_reading_ease = textstat.flesch_reading_ease(sentence_text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(sentence_text)
        gunning_fog = textstat.gunning_fog(sentence_text)
    except Exception:
        flesch_reading_ease = 0.0
        flesch_kincaid_grade = 0.0
        gunning_fog = 0.0
        
    # 5. Pronoun ratios (Inspired by Genre_Classifier)
    pron_1st_count = len(PRON_1ST.findall(sentence_text))
    pron_2nd_count = len(PRON_2ND.findall(sentence_text))
    pron_3rd_count = len(PRON_3RD.findall(sentence_text))
    
    pron_1st_ratio = pron_1st_count / max(1, word_count)
    pron_2nd_ratio = pron_2nd_count / max(1, word_count)
    pron_3rd_ratio = pron_3rd_count / max(1, word_count)
    
    # 6. Temporal & Tense features (Verb tags VBD/VBN vs VBP/VBZ/VBG/VB)
    tags = [token.tag_ for token in doc]
    past_verbs = sum(1 for t in tags if t in ("VBD", "VBN"))
    present_verbs = sum(1 for t in tags if t in ("VBP", "VBZ", "VBG", "VB"))
    past_tense_ratio = past_verbs / total_tokens
    present_tense_ratio = present_verbs / total_tokens
    
    # 7. Expanded Punctuation & Structural patterns
    commas = sentence_text.count(",")
    periods = sentence_text.count(".")
    exclamations = sentence_text.count("!")
    questions = sentence_text.count("?")
    semicolons = sentence_text.count(";")
    colons = sentence_text.count(":")
    parentheses = sentence_text.count("(") + sentence_text.count(")")
    
    number_tokens = sum(1 for token in doc if token.pos_ == "NUM" or token.is_digit)
    number_ratio = number_tokens / total_tokens
    
    # 8. Discourse markers
    causal_count = len(CAUSAL_MARKERS.findall(sentence_text))
    contrast_count = len(CONTRAST_MARKERS.findall(sentence_text))
    addition_count = len(ADDITION_MARKERS.findall(sentence_text))
    
    # 9. Syntactic complexity (Parse tree depth & dependency distance)
    avg_dep_distance = 0.0
    max_parse_depth = 0.0
    
    has_parser = "parser" in nlp.pipe_names
    if has_parser and len(doc) > 0:
        dep_distances = []
        depths = []
        for token in doc:
            dep_distances.append(abs(token.i - token.head.i))
            
            # Trace tree depth for token
            depth = 0
            curr = token
            visited = set()
            while curr.head != curr and curr not in visited:
                visited.add(curr)
                depth += 1
                curr = curr.head
            depths.append(depth)
            
        avg_dep_distance = sum(dep_distances) / len(doc)
        max_parse_depth = max(depths) if depths else 0.0
        
    features = {
        "word_count": float(word_count),
        "char_count": float(char_count),
        "avg_word_length": avg_word_length,
        "ttr": ttr,
        "stopword_ratio": stopword_ratio,
        "comma_count": float(commas),
        "period_count": float(periods),
        "exclamation_count": float(exclamations),
        "question_count": float(questions),
        "semicolon_count": float(semicolons),
        "colon_count": float(colons),
        "parenthesis_count": float(parentheses),
        "cap_ratio": cap_ratio,
        "title_ratio": title_ratio,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "gunning_fog": gunning_fog,
        "pron_1st_ratio": pron_1st_ratio,
        "pron_2nd_ratio": pron_2nd_ratio,
        "pron_3rd_ratio": pron_3rd_ratio,
        "past_tense_ratio": past_tense_ratio,
        "present_tense_ratio": present_tense_ratio,
        "number_ratio": number_ratio,
        "discourse_causal_count": float(causal_count),
        "discourse_contrast_count": float(contrast_count),
        "discourse_addition_count": float(addition_count),
        "avg_dep_distance": avg_dep_distance,
        "max_parse_depth": max_parse_depth
    }
    
    # Combine POS ratios
    features.update(pos_ratios)
    return features
