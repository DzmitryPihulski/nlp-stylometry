import numpy as np
import pandas as pd
import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.stats import mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

# ============================================================
# HERBERT
# ============================================================

MODEL_NAME = "allegro/herbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

herbert = AutoModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"

herbert = herbert.to(device)
herbert.eval()

print(f"HerBERT działa na: {device}")

# ============================================================
# HERBERT EMBEDDINGS
# ============================================================


def get_herbert_embeddings(
    texts,
    batch_size=16,
):
    """
    Mean pooling embeddingów HerBERT.
    """

    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = herbert(**encoded)

        token_emb = out.last_hidden_state

        mask = encoded["attention_mask"].unsqueeze(-1).float()

        emb = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        all_embs.append(emb.cpu().numpy())

        if i % 160 == 0:
            print(f"  HerBERT: {i}/{len(texts)}")

    return np.vstack(all_embs)


# ============================================================
# DOC2VEC
# ============================================================


def get_doc2vec_embeddings(
    texts,
    vector_size=300,
    window=10,
    min_count=2,
    epochs=40,
    workers=4,
):
    """
    Generuje embeddingi Doc2Vec.
    """

    tagged_docs = [
        TaggedDocument(words=text.split(), tags=[str(i)])
        for i, text in enumerate(texts)
    ]

    model = Doc2Vec(
        documents=tagged_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        dm=1,
        seed=42,
    )

    vectors = np.array([model.dv[str(i)] for i in range(len(tagged_docs))])

    return vectors


# ============================================================
# SIMILARITIES
# ============================================================


def compute_similarity_matrix(
    vectors,
    is_sparse=False,
):
    """
    Oblicza macierz cosine similarity.
    """

    if is_sparse:
        sim_matrix = cosine_similarity(vectors)

    else:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        norms = np.where(norms == 0, 1e-9, norms)

        vn = vectors / norms

        sim_matrix = vn @ vn.T

    return sim_matrix


def pairwise_similarities(
    vectors,
    labels,
    is_sparse=False,
):
    """
    Zwraca:
    - within-author similarities
    - between-author similarities
    """

    labels = np.array(labels)

    sim_matrix = compute_similarity_matrix(vectors, is_sparse)

    n = len(labels)

    rows, cols = np.triu_indices(n, k=1)

    sims = sim_matrix[rows, cols]

    same = labels[rows] == labels[cols]

    return sims[same], sims[~same]


# ============================================================
# STATISTICS
# ============================================================


def run_stats(
    within,
    between,
    name,
):
    """
    Mann-Whitney U:
    H0: within == between
    """

    u, p = mannwhitneyu(
        within,
        between,
        alternative="two-sided",
    )

    n1 = len(within)
    n2 = len(between)

    effect_r = 1 - (2 * u) / (n1 * n2)

    return {
        "reprezentacja": name,
        "within_mean": np.mean(within),
        "within_std": np.std(within),
        "between_mean": np.mean(between),
        "between_std": np.std(between),
        "n_within": n1,
        "n_between": n2,
        "U": u,
        "p_value": p,
        "effect_r": effect_r,
    }


# ============================================================
# AUTHOR PAIR ANALYSIS
# ============================================================


def mean_pairwise_per_author_pair(
    vectors,
    labels,
    authors,
    is_sparse=False,
):
    """
    Średnie similarity dla par autorów.
    """

    labels = np.array(labels)

    sim_matrix = compute_similarity_matrix(vectors, is_sparse)

    n = len(labels)

    rows, cols = np.triu_indices(n, k=1)

    result = {}

    for a1 in authors:
        for a2 in authors:
            mask = ((labels[rows] == a1) & (labels[cols] == a2)) | (
                (labels[rows] == a2) & (labels[cols] == a1)
            )

            if mask.sum() > 0:
                result[(a1, a2)] = sim_matrix[rows[mask], cols[mask]].mean()

            else:
                result[(a1, a2)] = np.nan

    return result


def pairs_to_matrix(
    pair_dict,
    authors,
):
    """
    Zamienia słownik par autorów na macierz.
    """

    n = len(authors)

    mat = np.full((n, n), np.nan)

    for i, a1 in enumerate(authors):
        for j, a2 in enumerate(authors):
            key = (a1, a2) if (a1, a2) in pair_dict else (a2, a1)

            mat[i, j] = pair_dict.get(key, np.nan)

    return mat
