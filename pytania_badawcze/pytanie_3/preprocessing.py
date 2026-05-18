from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc, Span

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


def load_nlp(model_name: str = "pl_core_news_sm"):
    """
    Ładuje pipeline spaCy dla języka polskiego.

    Wymagane są co najmniej:
    - tokenizacja
    - POS tagging
    - lematyzacja
    - segmentacja zdań

    Jeśli model nie ma parsera ani senter, dodawany jest sentencizer.
    """
    nlp = spacy.load(model_name)

    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    return nlp


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def iter_alpha_chunks(doc: Doc, chunk_size: int = 200) -> Iterator[tuple[int, Span]]:
    """
    Dzieli dokument na fragmenty po chunk_size tokenów alfabetycznych.

    Zwraca spany spaCy obejmujące pełny oryginalny kontekst tokenów
    między pierwszym i ostatnim tokenem danego fragmentu.
    """
    alpha_token_idxs = [t.i for t in doc if t.is_alpha]
    n_full_chunks = len(alpha_token_idxs) // chunk_size

    for chunk_no in range(n_full_chunks):
        idxs = alpha_token_idxs[chunk_no * chunk_size : (chunk_no + 1) * chunk_size]
        if not idxs:
            continue

        start_i = idxs[0]
        end_i = idxs[-1] + 1
        yield chunk_no, doc[start_i:end_i]


def extract_features(span: Span) -> dict:
    """
    Wyodrębnia cechy leksykalne i składniowe z jednego fragmentu tekstu.
    """
    alpha_tokens = [t for t in span if t.is_alpha]
    n = len(alpha_tokens)

    if n == 0:
        return {}

    words_lower = [t.text.lower() for t in alpha_tokens]
    lemmas = [t.lemma_.lower() if t.lemma_ else t.text.lower() for t in alpha_tokens]
    pos_tags = [t.pos_ for t in alpha_tokens]

    pos_counter = Counter(pos_tags)
    lemma_counter = Counter(lemmas)
    word_counter = Counter(words_lower)

    feats: dict[str, float] = {}

    # ---------- LEXICAL ----------
    feats["lex_avg_word_len"] = float(np.mean([len(w) for w in words_lower]))
    feats["lex_ttr"] = _safe_div(len(set(lemmas)), n)
    feats["lex_hapax_ratio"] = _safe_div(
        sum(1 for v in lemma_counter.values() if v == 1), n
    )

    for fw in FUNCTION_WORDS:
        feats[f"lex_fw_{fw}"] = _safe_div(word_counter.get(fw, 0), n)

    # ---------- SYNTACTIC / MORPHOSYNTACTIC ----------
    for pos in POS_TAGS:
        feats[f"syn_pos_{pos}"] = _safe_div(pos_counter.get(pos, 0), n)

    n_verb = pos_counter.get("VERB", 0)
    n_noun = pos_counter.get("NOUN", 0)
    n_adj = pos_counter.get("ADJ", 0)

    feats["syn_verb_noun_ratio"] = _safe_div(n_verb, n_noun)
    feats["syn_adj_noun_ratio"] = _safe_div(n_adj, n_noun)

    # Pozycja czasownika w zdaniu:
    # 0 = początek zdania, 1 = koniec zdania.
    sent_verb_positions = []
    sent_verb_initial = []
    sent_verb_final = []
    sent_lengths = []

    try:
        sentences = list(span.sents)
    except Exception:
        sentences = []

    for sent in sentences:
        sent_alpha = [t for t in sent if t.is_alpha]
        if not sent_alpha:
            continue

        sent_lengths.append(len(sent_alpha))

        denom = max(len(sent_alpha) - 1, 1)
        verb_positions = [
            i / denom for i, tok in enumerate(sent_alpha) if tok.pos_ == "VERB"
        ]

        if verb_positions:
            sent_verb_positions.extend(verb_positions)

        sent_verb_initial.append(1.0 if sent_alpha[0].pos_ == "VERB" else 0.0)
        sent_verb_final.append(1.0 if sent_alpha[-1].pos_ == "VERB" else 0.0)

    feats["syn_avg_sentence_len"] = (
        float(np.mean(sent_lengths)) if sent_lengths else 0.0
    )
    feats["syn_verb_pos_mean"] = (
        float(np.mean(sent_verb_positions)) if sent_verb_positions else 0.5
    )
    feats["syn_verb_pos_std"] = (
        float(np.std(sent_verb_positions)) if len(sent_verb_positions) > 1 else 0.0
    )
    feats["syn_verb_sentence_initial_rate"] = (
        float(np.mean(sent_verb_initial)) if sent_verb_initial else 0.0
    )
    feats["syn_verb_sentence_final_rate"] = (
        float(np.mean(sent_verb_final)) if sent_verb_final else 0.0
    )

    return feats


def build_feature_dataset(
    df: pd.DataFrame,
    chunk_size: int = 200,
    nlp=None,
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Buduje zbiór cech z oryginalnego df.

    Każdy wiersz wynikowy to jeden fragment o długości chunk_size
    tokenów alfabetycznych.
    """
    required_cols = {"author", "kind", "genre", "title", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn w df: {sorted(missing)}")

    if nlp is None:
        nlp = load_nlp()

    records = df.reset_index(drop=True).copy()
    records["text"] = records["text"].fillna("").astype(str)

    rows = []

    texts = records["text"].tolist()

    for row_idx, (doc, row) in enumerate(
        zip(nlp.pipe(texts, batch_size=batch_size), records.itertuples(index=False))
    ):
        if row_idx % 10 == 0:
            print(f"  spaCy: {row_idx}/{len(records)} wierszy...")

        meta = {
            "author": row.author,
            "kind": row.kind,
            "genre": row.genre,
            "title": row.title,
        }

        # grupowanie na poziomie oryginalnego utworu / wariantu tekstu
        work_id = f"{row.author}||{row.kind}||{row.genre}||{row.title}"

        for chunk_no, span in iter_alpha_chunks(doc, chunk_size=chunk_size):
            feats = extract_features(span)
            if not feats:
                continue

            feats.update(meta)
            feats["work_id"] = work_id
            feats["chunk_id"] = chunk_no
            feats["n_alpha_tokens"] = sum(1 for t in span if t.is_alpha)
            rows.append(feats)

    return pd.DataFrame(rows)
