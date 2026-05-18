# evaluation.py

import warnings

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURACJA
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

AUTHORS = [
    "Adam_Mickiewicz",
    "Juliusz_Słowacki",
    "Zygmunt_Krasiński",
]

KINDS = ["Dramat", "Epika"]

# ============================================================
# PIPELINE
# ============================================================


def make_pipeline(vectorizer) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "svm",
                LinearSVC(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# WARIANT 1 — BEZ KONTROLI GATUNKU
# ============================================================


def evaluate_no_genre_control(
    df: pd.DataFrame,
    vectorizer,
) -> dict:
    """
    Klasyfikacja autora bez kontroli gatunku.

    - wszystkie gatunki razem
    - GroupKFold po title
    - brak wycieku między fragmentami tego samego utworu
    """

    X = df["text"].values
    y = df["author"].values
    groups = df["title"].values

    gkf = GroupKFold(n_splits=N_SPLITS)

    f1s = []
    accs = []
    gmeans = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups),
        start=1,
    ):
        clf = make_pipeline(vectorizer)

        clf.fit(X[train_idx], y[train_idx])

        pred = clf.predict(X[test_idx])

        f1 = f1_score(
            y[test_idx],
            pred,
            average="macro",
            zero_division=0,
        )

        acc = accuracy_score(
            y[test_idx],
            pred,
        )

        gmean = geometric_mean_score(
            y[test_idx],
            pred,
            average="macro",
        )

        f1s.append(f1)
        accs.append(acc)
        gmeans.append(gmean)

    return {
        "f1": np.array(f1s),
        "acc": np.array(accs),
        "gmean": np.array(gmeans),
    }


# ============================================================
# WARIANT 2 — Z KONTROLĄ GATUNKU
# ============================================================


def evaluate_with_genre_control(
    df: pd.DataFrame,
    vectorizer,
) -> dict:
    """
    Kontrola gatunku:
    - osobny model dla każdego gatunku
    - train/test WYŁĄCZNIE w obrębie danego gatunku
    - GroupKFold po title
    """

    global_f1 = []
    global_acc = []
    global_gmean = []

    per_genre = []

    for genre in sorted(df["kind"].unique()):
        sub = df[df["kind"] == genre].copy()

        # minimum 2 autorów
        if sub["author"].nunique() < 2:
            print(f"[SKIP] {genre}: tylko {sub['author'].nunique()} autor.")
            continue

        # liczba grup = liczba tytułów
        n_titles = sub["title"].nunique()

        n_splits_actual = min(
            N_SPLITS,
            n_titles,
        )

        if n_splits_actual < 2:
            print(f"[SKIP] {genre}: za mało tytułów.")
            continue

        X = sub["text"].values
        y = sub["author"].values
        groups = sub["title"].values

        gkf = GroupKFold(n_splits=n_splits_actual)

        genre_f1 = []
        genre_acc = []
        genre_gmean = []

        for fold, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups),
            start=1,
        ):
            clf = make_pipeline(vectorizer)

            clf.fit(
                X[train_idx],
                y[train_idx],
            )

            pred = clf.predict(X[test_idx])

            f1 = f1_score(
                y[test_idx],
                pred,
                average="macro",
                zero_division=0,
            )

            acc = accuracy_score(
                y[test_idx],
                pred,
            )

            gmean = geometric_mean_score(
                y[test_idx],
                pred,
                average="macro",
            )

            genre_f1.append(f1)
            genre_acc.append(acc)
            genre_gmean.append(gmean)

        # agregacja globalna
        global_f1.extend(genre_f1)
        global_acc.extend(genre_acc)
        global_gmean.extend(genre_gmean)

        # raport per gatunek
        per_genre.append(
            {
                "genre": genre,
                "authors": sorted(sub["author"].unique()),
                "n_chunks": len(sub),
                "n_titles": n_titles,
                "n_folds": n_splits_actual,
                "f1_mean": np.mean(genre_f1),
                "f1_std": np.std(genre_f1),
                "acc_mean": np.mean(genre_acc),
                "acc_std": np.std(genre_acc),
                "gmean_mean": np.mean(genre_gmean),
                "gmean_std": np.std(genre_gmean),
            }
        )

    return {
        "f1": np.array(global_f1),
        "acc": np.array(global_acc),
        "gmean": np.array(global_gmean),
        "per_genre": per_genre,
    }


# ============================================================
# TESTY STATYSTYCZNE
# ============================================================


def statistical_tests(
    no_control_scores: np.ndarray,
    genre_control_scores: np.ndarray,
) -> dict:
    """
    Porównanie wariantów:
    - test sparowany
    - porównujemy średnie foldów

    Ponieważ liczba obserwacji może być różna:
    - redukujemy do wspólnego minimum
    """

    n = min(
        len(no_control_scores),
        len(genre_control_scores),
    )

    x = no_control_scores[:n]
    y = genre_control_scores[:n]

    # t-test sparowany
    t_stat, t_p = ttest_rel(
        x,
        y,
    )

    # Wilcoxon
    try:
        w_stat, w_p = wilcoxon(
            x,
            y,
        )
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    return {
        "n": n,
        "mean_no": np.mean(x),
        "std_no": np.std(x),
        "mean_gen": np.mean(y),
        "std_gen": np.std(y),
        "delta": np.mean(y) - np.mean(x),
        "ttest_t": t_stat,
        "ttest_p": t_p,
        "wilcoxon_w": w_stat,
        "wilcoxon_p": w_p,
        "significant": (t_p < 0.05 and w_p < 0.05),
    }


# ============================================================
# PEŁNE URUCHOMIENIE
# ============================================================


def run_experiment(
    df_chunks: pd.DataFrame,
    vectorizer,
    vectorizer_name: str,
) -> dict:

    print("\n" + "=" * 60)
    print(f"Wektoryzacja: {vectorizer_name}")
    print("=" * 60)

    # ----------------------------------------
    # wariant bez kontroli
    # ----------------------------------------

    no_control = evaluate_no_genre_control(
        df_chunks,
        vectorizer,
    )

    # ----------------------------------------
    # wariant z kontrolą
    # ----------------------------------------

    genre_control = evaluate_with_genre_control(
        df_chunks,
        vectorizer,
    )

    # ----------------------------------------
    # statystyka
    # ----------------------------------------

    stats_f1 = statistical_tests(
        no_control["f1"],
        genre_control["f1"],
    )

    stats_gmean = statistical_tests(
        no_control["gmean"],
        genre_control["gmean"],
    )

    return {
        "vectorizer": vectorizer_name,
        # bez kontroli
        "f1_no": no_control["f1"],
        "acc_no": no_control["acc"],
        "gmean_no": no_control["gmean"],
        # z kontrolą
        "f1_gen": genre_control["f1"],
        "acc_gen": genre_control["acc"],
        "gmean_gen": genre_control["gmean"],
        # szczegóły
        "per_genre": genre_control["per_genre"],
        # testy
        "stats_f1": stats_f1,
        "stats_gmean": stats_gmean,
    }


# ============================================================
# RAPORT
# ============================================================


def print_report(result: dict):

    print("\n" + "=" * 60)
    print("WYNIKI")
    print("=" * 60)

    print(f"\nWektoryzacja: {result['vectorizer']}")

    # ========================================================
    # F1
    # ========================================================

    s = result["stats_f1"]

    print("\n[F1-score macro]")

    print(f"Bez kontroli gatunku: {s['mean_no']:.4f} ± {s['std_no']:.4f}")

    print(f"Z kontrolą gatunku: {s['mean_gen']:.4f} ± {s['std_gen']:.4f}")

    print(f"Delta: {s['delta']:+.4f}")

    print(f"Paired t-test: t={s['ttest_t']:.4f}, p={s['ttest_p']:.4f}")

    print(f"Wilcoxon: W={s['wilcoxon_w']:.4f}, p={s['wilcoxon_p']:.4f}")

    # ========================================================
    # G-MEAN
    # ========================================================

    s = result["stats_gmean"]

    print("\n[Geometric Mean]")

    print(f"Bez kontroli gatunku: {s['mean_no']:.4f} ± {s['std_no']:.4f}")

    print(f"Z kontrolą gatunku: {s['mean_gen']:.4f} ± {s['std_gen']:.4f}")

    print(f"Delta: {s['delta']:+.4f}")

    print(f"Paired t-test: t={s['ttest_t']:.4f}, p={s['ttest_p']:.4f}")

    print(f"Wilcoxon: W={s['wilcoxon_w']:.4f}, p={s['wilcoxon_p']:.4f}")

    # ========================================================
    # PER GATUNEK
    # ========================================================

    print("\nWyniki per gatunek:")

    for g in result["per_genre"]:
        print(f"\n{g['genre']}")

        print(f"  Autorzy: {', '.join(g['authors'])}")

        print(f"  Chunks: {g['n_chunks']}")

        print(f"  F1: {g['f1_mean']:.4f} ± {g['f1_std']:.4f}")

        print(f"  G-Mean: {g['gmean_mean']:.4f} ± {g['gmean_std']:.4f}")
