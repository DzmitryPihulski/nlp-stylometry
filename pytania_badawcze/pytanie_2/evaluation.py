import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================
# B. REPREZENTACJA 2: HerBERT embeddings
# ============================================================
import torch
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "allegro/herbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
herbert = AutoModel.from_pretrained(MODEL_NAME)
herbert.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
herbert = herbert.to(device)
print(f"HerBERT działa na: {device}")


def get_herbert_embeddings(texts, batch_size=16):
    """Mean-pooling ostatniej warstwy ukrytej HerBERTa."""
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

        # mean pooling z uwzględnieniem attention mask
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        token_emb = out.last_hidden_state
        emb = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        all_embs.append(emb.cpu().numpy())

        if i % 160 == 0:
            print(f"  HerBERT: {i}/{len(texts)} fragmentów...")

    return np.vstack(all_embs)


def pairwise_similarities(vectors, labels, is_sparse=False):
    """
    Zwraca (within, between) – tablice cosine similarity
    dla par: ten-sam-autor i różni-autorzy.
    Używa numpy/sklearn – bez pętli po parach.
    """
    labels = np.array(labels)
    n = len(labels)

    if is_sparse:
        sim_matrix = cosine_similarity(vectors)  # dense (n x n)
    else:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        vn = vectors / norms
        sim_matrix = vn @ vn.T  # dense (n x n)

    rows, cols = np.triu_indices(n, k=1)  # górny trójkąt
    sims = sim_matrix[rows, cols]
    same = labels[rows] == labels[cols]

    return sims[same], sims[~same]


def run_stats(within, between, name):
    """Mann-Whitney U: H0 – within == between."""
    u, p = mannwhitneyu(within, between, alternative="two-sided")

    # rank-biserial correlation (effect size)
    n1, n2 = len(within), len(between)
    r_rb = 1 - (2 * u) / (n1 * n2)

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
        "effect_r": r_rb,
    }


def mean_pairwise_per_author_pair(vectors, labels, authors, is_sparse=False):
    """Średnie podobieństwo dla każdej pary (A,B), A<=B."""
    labels = np.array(labels)
    n = len(labels)

    if is_sparse:
        sim_matrix = cosine_similarity(vectors)
    else:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        vn = vectors / norms
        sim_matrix = vn @ vn.T

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


def pairs_to_matrix(pair_dict, authors):
    n = len(authors)
    mat = np.full((n, n), np.nan)
    for i, a1 in enumerate(authors):
        for j, a2 in enumerate(authors):
            key = (a1, a2) if (a1, a2) in pair_dict else (a2, a1)
            mat[i, j] = pair_dict.get(key, np.nan)
    return mat
