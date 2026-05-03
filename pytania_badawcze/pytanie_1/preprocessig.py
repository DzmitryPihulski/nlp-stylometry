import re

import pandas as pd
import spacy
from spacy.lang.pl.stop_words import STOP_WORDS

CHUNK_SIZE = 200
# ============================================================
# 2. PREPROCESSING
# ============================================================

nlp = spacy.load("pl_core_news_sm", disable=["parser", "ner"])


def preprocess(text: str) -> list[str]:
    """Tokenizacja + lematyzacja + usunięcie stop words i niealfabetycznych tokenów."""
    text = text.lower()
    text = re.sub(r"[^a-ząćęłńóśźż\s]", " ", text)
    doc = nlp(text)
    return [
        token.lemma_
        for token in doc
        if token.is_alpha and token.lemma_ not in STOP_WORDS
    ]


def split_into_chunks(row: pd.Series, chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """
    Dzieli jeden wiersz (jeden tekst źródłowy) na fragmenty po `chunk_size` słów.
    Fragmenty krótsze niż chunk_size są odrzucane (brzegowe).
    """
    tokens = preprocess(row["text"])
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i : i + chunk_size]
        if len(chunk) < chunk_size:
            continue
        chunks.append(
            {
                "author": row["author"],
                "kind": row["kind"],
                "title": row["title"],
                "text": " ".join(chunk),
            }
        )
    return chunks


def build_chunk_dataset(df: pd.DataFrame) -> pd.DataFrame:
    all_chunks = []
    for _, row in df.iterrows():
        all_chunks.extend(split_into_chunks(row))
    return pd.DataFrame(all_chunks)
