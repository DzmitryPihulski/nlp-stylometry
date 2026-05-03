# ============================================================
# PYTANIE 3: Cechy leksykalne vs składniowe w klasyfikacji autora
# ============================================================

import re
from collections import Counter

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from spacy.lang.pl.stop_words import STOP_WORDS

nlp = spacy.load("pl_core_news_sm")  # pełny pipeline: tagger + morph

# ============================================================
# A. DEFINICJA CECH
# ============================================================

# Słowa funkcyjne do zliczania (przed lematyzacją / usunięciem stop words)
FUNCTION_WORDS = [
    "i",
    "że",
    "ale",
    "a",
    "bo",
    "lecz",
    "gdy",
    "jak",
    "czy",
    "się",
    "nie",
    "już",
    "też",
    "więc",
    "jednak",
    "tylko",
    "jeszcze",
    "tam",
    "tu",
    "o",
    "z",
    "w",
    "na",
    "do",
    "po",
]

# Tagi POS używane w cechach składniowych
POS_TAGS = [
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "ADP",
    "CCONJ",
    "SCONJ",
    "PART",
    "NUM",
]


# ============================================================
# B. EKSTRAKCJA CECH Z JEDNEGO FRAGMENTU (200 słów)
# ============================================================


def extract_features(chunk_doc_tokens: list) -> dict:
    """
    chunk_doc_tokens: lista spacy Token z jednego 200-słowowego fragmentu
    Zwraca słownik {nazwa_cechy: wartość}.
    """
    alpha_tokens = [t for t in chunk_doc_tokens if t.is_alpha]
    n = len(alpha_tokens)
    if n == 0:
        return {}

    words_lower = [t.text.lower() for t in alpha_tokens]
    lemmas = [t.lemma_.lower() for t in alpha_tokens]
    pos_tags_tok = [t.pos_ for t in alpha_tokens]
    pos_counter = Counter(pos_tags_tok)

    feats = {}

    # ── LEKSYKALNE ───────────────────────────────────────────
    # 1. Średnia długość słowa (znaki)
    feats["lex_avg_word_len"] = np.mean([len(w) for w in words_lower])

    # 2. Type-Token Ratio (bogactwo leksykalne)
    feats["lex_ttr"] = len(set(lemmas)) / n

    # 3. Hapax legomena ratio
    freq = Counter(lemmas)
    feats["lex_hapax_ratio"] = sum(1 for v in freq.values() if v == 1) / n

    # 4. Częstości słów funkcyjnych
    for fw in FUNCTION_WORDS:
        feats[f"lex_fw_{fw}"] = words_lower.count(fw) / n

    # ── SKŁADNIOWE ────────────────────────────────────────────
    # 5. Procentowy udział każdej części mowy
    for pos in POS_TAGS:
        feats[f"syn_pos_{pos}"] = pos_counter.get(pos, 0) / n

    # 6. Stosunek czasowników do rzeczowników
    n_verb = pos_counter.get("VERB", 0)
    n_noun = pos_counter.get("NOUN", 0)
    feats["syn_verb_noun_ratio"] = n_verb / (n_noun + 1e-9)

    # 7. Pozycja czasowników w zdaniu (względna, 0=początek, 1=koniec)
    verb_pos = [i / n for i, t in enumerate(alpha_tokens) if t.pos_ == "VERB"]
    feats["syn_verb_pos_mean"] = np.mean(verb_pos) if verb_pos else 0.5
    feats["syn_verb_pos_std"] = np.std(verb_pos) if len(verb_pos) > 1 else 0.0
    feats["syn_verb_early_r"] = sum(1 for vp in verb_pos if vp < 0.33) / (
        len(verb_pos) + 1e-9
    )
    feats["syn_verb_late_r"] = sum(1 for vp in verb_pos if vp > 0.67) / (
        len(verb_pos) + 1e-9
    )

    # 8. Przymiotniki/rzeczowniki (nasycenie opisowe)
    n_adj = pos_counter.get("ADJ", 0)
    feats["syn_adj_noun_ratio"] = n_adj / (n_noun + 1e-9)

    return feats


# ============================================================
# C. BUDOWA DATASETU CECH (z ORYGINALNEGO df, nie df_chunks)
# ============================================================


def build_feature_dataset(df, chunk_size=200):
    """
    Przetwarza każdy wiersz df przez spacy, dzieli na
    fragmenty po chunk_size słów alpha i wyciąga cechy.
    """
    rows = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 10 == 0:
            print(f"  Spacy: {idx}/{total} wierszy...")

        doc = nlp(row["text"])
        alpha_tokens = [t for t in doc if t.is_alpha]

        for start in range(0, len(alpha_tokens), chunk_size):
            chunk = alpha_tokens[start : start + chunk_size]
            if len(chunk) < chunk_size:
                continue

            feats = extract_features(chunk)
            if not feats:
                continue

            feats["author"] = row["author"]
            feats["kind"] = row["kind"]
            feats["title"] = row["title"]
            rows.append(feats)

    return pd.DataFrame(rows)
