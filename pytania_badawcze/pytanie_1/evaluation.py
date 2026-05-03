import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


RANDOM_STATE = 42
N_SPLITS = 5
AUTHORS = ["Adam_Mickiewicz", "Juliusz_Słowacki", "Zygmunt_Krasiński"]
KINDS = ["Dramat", "Epika"]


def make_pipeline(vectorizer) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("svm", LinearSVC(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def evaluate_no_genre_control(
    df: pd.DataFrame, vectorizer
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wariant 1: BEZ kontroli gatunku.
    GroupKFold po tytule zapewnia brak wycieku danych między tytułami.
    Zwraca tablice F1 i Accuracy dla każdego foldu.
    """
    X = df["text"].values
    y = df["author"].values
    groups = df["title"].values

    gkf = GroupKFold(n_splits=N_SPLITS)
    f1s, accs = [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = make_pipeline(vectorizer)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
        accs.append(accuracy_score(y[test_idx], pred))

    return np.array(f1s), np.array(accs)


def evaluate_with_genre_control(
    df: pd.DataFrame, vectorizer
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Wariant 2: Z KONTROLĄ GATUNKU.
    Dla każdego gatunku osobny pipeline: trenowanie i testowanie w obrębie
    tego samego gatunku. Gatunki z < 2 autorami są pomijane.

    Zwraca:
      - tablice F1 i Accuracy (zagregowane po wszystkich gatunkach i foldach)
      - listę szczegółowych wyników per gatunek
    """
    f1s, accs = [], []
    per_genre = []

    for genre in sorted(df["kind"].unique()):
        sub = df[df["kind"] == genre]

        # Wymóg: co najmniej 2 autorów w gatunku
        if sub["author"].nunique() < 2:
            print(
                f"  [SKIP] Gatunek '{genre}': tylko {sub['author'].nunique()} autor(zy) — pomijam."
            )
            continue

        # Wymóg: co najmniej N_SPLITS tytułów (grup)
        n_titles = sub["title"].nunique()
        n_splits_actual = min(N_SPLITS, n_titles)
        if n_splits_actual < 2:
            print(
                f"  [SKIP] Gatunek '{genre}': za mało tytułów ({n_titles}) — pomijam."
            )
            continue

        X = sub["text"].values
        y = sub["author"].values
        groups = sub["title"].values

        gkf = GroupKFold(n_splits=n_splits_actual)
        genre_f1s, genre_accs = [], []

        for train_idx, test_idx in gkf.split(X, y, groups):
            clf = make_pipeline(vectorizer)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            genre_f1s.append(
                f1_score(y[test_idx], pred, average="macro", zero_division=0)
            )
            genre_accs.append(accuracy_score(y[test_idx], pred))

        f1s.extend(genre_f1s)
        accs.extend(genre_accs)

        per_genre.append(
            {
                "genre": genre,
                "f1_mean": np.mean(genre_f1s),
                "f1_std": np.std(genre_f1s),
                "acc_mean": np.mean(genre_accs),
                "acc_std": np.std(genre_accs),
                "n_folds": n_splits_actual,
                "n_chunks": len(sub),
                "authors": list(sub["author"].unique()),
            }
        )

    return np.array(f1s), np.array(accs), per_genre


# ============================================================
# 6. TESTY STATYSTYCZNE
# ============================================================


def statistical_tests(f1_no: np.ndarray, f1_gen: np.ndarray) -> dict:
    """
    Testy dla prób NIEZALEŻNYCH:
      - Welch t-test (nierówne wariancje, nierówne liczebności)
      - Mann-Whitney U (nieparametryczny odpowiednik)

    UWAGA: Próby są niezależne, ponieważ f1_no i f1_gen mają różne długości
    (5 foldów vs 5 × liczba_gatunków), co wyklucza testy sparowane.
    """
    t_stat, t_p = ttest_ind(f1_no, f1_gen, equal_var=False)

    try:
        u_stat, u_p = mannwhitneyu(f1_no, f1_gen, alternative="two-sided")
    except ValueError:
        u_stat, u_p = np.nan, np.nan

    return {
        "n_no": len(f1_no),
        "n_gen": len(f1_gen),
        "f1_no_mean": np.mean(f1_no),
        "f1_no_std": np.std(f1_no),
        "f1_gen_mean": np.mean(f1_gen),
        "f1_gen_std": np.std(f1_gen),
        "delta": np.mean(f1_gen) - np.mean(f1_no),
        "welch_t": t_stat,
        "welch_p": t_p,
        "mwu_u": u_stat,
        "mwu_p": u_p,
        "significant": t_p < 0.05 and u_p < 0.05,
    }
