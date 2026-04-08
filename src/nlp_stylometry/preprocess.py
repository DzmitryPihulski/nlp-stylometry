"""Preprocessing pipeline for stylometry data."""

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Corpus:
    author: str
    chunks: list[str]  # each chunk is N sentences joined


def _is_polish(text: str, min_polish_ratio: float = 0.005) -> bool:
    """Heuristic: Polish texts have Polish diacritics at a detectable rate."""
    polish_chars = set("ąęóśźżćńłĄĘÓŚŹŻĆŃŁ")
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    ratio = sum(1 for c in alpha if c in polish_chars) / len(alpha)
    return ratio >= min_polish_ratio


def _remove_drama_markup(text: str) -> str:
    """Remove speaker labels and stage directions from drama texts.

    Patterns:
    - Speaker labels: a line consisting only of uppercase Polish letters/spaces
      (e.g. GUŚLARZ, CHÓR, STARZEC PIERWSZY)
    - Stage directions: lines enclosed in / ... /
    """
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Stage direction: / ... /
        if re.match(r"^/.*/$", stripped):
            continue
        # Speaker label: all-caps Polish word(s), optionally with dots/numbers
        if re.match(r"^[A-ZŁŚŻŹĆŃ][A-ZŁŚŻŹĆŃ\s\d.]*$", stripped) and len(stripped) > 1:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _split_sentences(text: str) -> list[str]:
    """Split Polish text into sentences."""
    # Split on sentence-ending punctuation followed by whitespace + uppercase
    raw = re.split(r"(?<=[.!?…])\s+(?=[A-ZŁŚŻŹĆŃA-Z«\"])", text)
    sentences = []
    for s in raw:
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences


def load_author(author_dir: Path, min_sentences: int = 5) -> list[str]:
    """Load, clean, and return all sentences for a single author.

    Files with fewer than `min_sentences` after cleaning are skipped.
    """
    sentences: list[str] = []
    for path in sorted(author_dir.iterdir()):
        if path.suffix != ".txt":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue  # skip empty files
        if not _is_polish(text):
            continue  # skip non-Polish files
        text = _remove_drama_markup(text)
        file_sentences = _split_sentences(text)
        if len(file_sentences) < min_sentences:
            continue  # skip fragments too short to be useful
        sentences.extend(file_sentences)
    return sentences


def build_corpus(
    data_dir: Path, chunk_size: int = 100, min_sentences: int = 5
) -> list[Corpus]:
    """Build balanced corpus chunks for each author.

    Each author ends up with the same number of chunks of `chunk_size` sentences.
    Files with fewer than `min_sentences` after cleaning are skipped.
    """
    author_sentences: dict[str, list[str]] = {}
    for author_dir in sorted(data_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        author = author_dir.name
        author_sentences[author] = load_author(author_dir, min_sentences=min_sentences)

    # Determine max chunks limited by smallest author
    chunks_per_author = min(
        len(sents) // chunk_size for sents in author_sentences.values()
    )

    corpora = []
    for author, sentences in author_sentences.items():
        chunks = [
            " ".join(sentences[i * chunk_size : (i + 1) * chunk_size])
            for i in range(chunks_per_author)
        ]
        corpora.append(Corpus(author=author, chunks=chunks))

    return corpora
