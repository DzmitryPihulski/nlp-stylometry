"""Preprocessing pipeline for stylometry data."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Corpus:
    author: str
    kind: str
    genre: str
    title: str
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
    raw = re.split(r"(?<=[.!?…])\s+(?=[A-ZŁŚŻŹĆŃA-Z«\"])", text)
    sentences = []
    for s in raw:
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences


def _read_title(sidecar: Path) -> str:
    """Read title from a JSON sidecar file. Returns 'unknown' if missing."""
    if not sidecar.exists():
        return "unknown"
    try:
        return json.loads(sidecar.read_text(encoding="utf-8")).get("title", "unknown")
    except (json.JSONDecodeError, OSError):
        return "unknown"


def build_corpus(
    data_dir: Path, chunk_size: int = 100, min_sentences: int = 5
) -> list[Corpus]:
    """Build balanced corpus chunks per (author, kind, genre, title).

    Directory layout expected:
        data_dir/{Author}/{Kind}/{Genre}/{idx}.txt
        data_dir/{Author}/{Kind}/{Genre}/{idx}.json   ← title sidecar

    Each text file is chunked independently so every chunk carries a single
    unambiguous title. The total chunk count is balanced per author — each
    author contributes the same number of chunks across all their works.

    Files with fewer than `min_sentences` after cleaning are skipped.
    """
    # Step 1: load sentences per file, tagged with (author, kind, genre, title)
    file_records: list[tuple[str, str, str, str, list[str]]] = []

    for author_dir in sorted(data_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        author = author_dir.name
        for kind_dir in sorted(author_dir.iterdir()):
            if not kind_dir.is_dir():
                continue
            kind = kind_dir.name
            for genre_dir in sorted(kind_dir.iterdir()):
                if not genre_dir.is_dir():
                    continue
                genre = genre_dir.name
                for txt_file in sorted(genre_dir.glob("*.txt")):
                    text = txt_file.read_text(encoding="utf-8").strip()
                    if not text or not _is_polish(text):
                        continue
                    text = _remove_drama_markup(text)
                    sents = _split_sentences(text)
                    if len(sents) < min_sentences:
                        continue
                    sidecar = txt_file.with_suffix(".json")
                    title = _read_title(sidecar)
                    file_records.append((author, kind, genre, title, sents))

    # Step 2: chunk within each file
    chunks_by_author: dict[str, list[tuple[str, str, str, str, str]]] = {}
    for author, kind, genre, title, sentences in file_records:
        n = len(sentences) // chunk_size
        for i in range(n):
            chunk = " ".join(sentences[i * chunk_size : (i + 1) * chunk_size])
            chunks_by_author.setdefault(author, []).append((author, kind, genre, title, chunk))

    if not chunks_by_author:
        return []

    # Step 3: balance — cap total chunks per author at the minimum across all authors
    min_chunks = min(len(chunks) for chunks in chunks_by_author.values())

    # Step 4: build Corpus objects grouped by (author, kind, genre, title)
    corpus_map: dict[tuple[str, str, str, str], list[str]] = {}
    for author, chunk_list in chunks_by_author.items():
        for _, kind, genre, title, text in chunk_list[:min_chunks]:
            corpus_map.setdefault((author, kind, genre, title), []).append(text)

    return [
        Corpus(author=key[0], kind=key[1], genre=key[2], title=key[3], chunks=chunk_texts)
        for key, chunk_texts in corpus_map.items()
    ]
