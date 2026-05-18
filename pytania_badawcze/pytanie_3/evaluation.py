from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    HAS_STRATIFIED_GROUP_KFOLD = False


def get_feature_columns(df_feats: pd.DataFrame):
    lex_cols = [c for c in df_feats.columns if c.startswith("lex_")]
    syn_cols = [c for c in df_feats.columns if c.startswith("syn_")]
    return lex_cols, syn_cols


def multiclass_gmean(y_true, y_pred, labels=None) -> float:
    """
    Geometric mean dla klasyfikacji wieloklasowej:
    gmean = (recall_1 * recall_2 * ... * recall_k)^(1/k)

    Jeśli któryś recall jest równy 0, wynik też jest 0.
    """
    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    recalls = np.asarray(recalls, dtype=float)

    if len(recalls) == 0:
        return 0.0

    if np.any(recalls <= 0):
        return 0.0

    return float(np.exp(np.mean(np.log(recalls))))


def _effective_n_splits(
    df_feats: pd.DataFrame,
    n_splits: int,
    group_col: str,
    target_col: str,
) -> int:
    """
    Dobiera liczbę foldów tak, aby była możliwa przy podziale grupowym.
    """
    if group_col not in df_feats.columns:
        raise ValueError(f"Brakuje kolumny group_col='{group_col}'")
    if target_col not in df_feats.columns:
        raise ValueError(f"Brakuje kolumny target_col='{target_col}'")

    # Liczba unikalnych grup w każdej klasie
    groups_per_class = (
        df_feats[[target_col, group_col]]
        .drop_duplicates()
        .groupby(target_col)[group_col]
        .nunique()
    )

    if groups_per_class.empty:
        raise ValueError(
            "Nie można wyznaczyć liczby foldów — brak danych po grupowaniu."
        )

    max_allowed = int(groups_per_class.min())
    n_unique_groups = int(df_feats[group_col].nunique())

    effective = min(int(n_splits), max_allowed, n_unique_groups)

    if effective < 2:
        raise ValueError(
            "Za mało grup w najmniejszej klasie, żeby zrobić sensowną walidację krzyżową."
        )

    return effective


def build_cv_splits(
    df_feats: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "work_id",
    target_col: str = "author",
    random_state: int = 42,
):
    """
    Buduje wspólne splity CV, których można użyć do porównania obu zestawów cech.
    """
    effective_n_splits = _effective_n_splits(
        df_feats=df_feats,
        n_splits=n_splits,
        group_col=group_col,
        target_col=target_col,
    )

    X = df_feats[
        [c for c in df_feats.columns if c.startswith("lex_") or c.startswith("syn_")]
    ].to_numpy(dtype=float)
    y = df_feats[target_col].to_numpy()
    groups = df_feats[group_col].to_numpy()

    if HAS_STRATIFIED_GROUP_KFOLD:
        splitter = StratifiedGroupKFold(
            n_splits=effective_n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = GroupKFold(n_splits=effective_n_splits)
        splits = list(splitter.split(X, y, groups))

    return splits, effective_n_splits


def evaluate_feature_set(
    df_feats: pd.DataFrame,
    feature_cols,
    splits,
    target_col: str = "author",
):
    """
    Ewaluacja LinearSVC na wybranym zestawie cech.
    Zwraca:
      f1_scores, acc_scores, gmean_scores
    """
    X = df_feats[feature_cols].to_numpy(dtype=float)
    y = df_feats[target_col].to_numpy()

    classes = np.unique(y)

    f1_list = []
    acc_list = []
    gmean_list = []

    for train_idx, test_idx in splits:
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", LinearSVC(max_iter=10000, class_weight="balanced")),
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(X[train_idx], y[train_idx])

        pred = clf.predict(X[test_idx])

        f1_list.append(
            f1_score(
                y[test_idx],
                pred,
                average="macro",
                labels=classes,
                zero_division=0,
            )
        )
        acc_list.append(accuracy_score(y[test_idx], pred))
        gmean_list.append(multiclass_gmean(y[test_idx], pred, labels=classes))

    return (
        np.asarray(f1_list, dtype=float),
        np.asarray(acc_list, dtype=float),
        np.asarray(gmean_list, dtype=float),
    )


def get_feature_importance(
    df_feats: pd.DataFrame,
    feature_cols,
    target_col: str = "author",
):
    """
    Trenuje model na całości i zwraca średnią z |wag| cech.
    """
    X = df_feats[feature_cols].to_numpy(dtype=float)
    y = df_feats[target_col].to_numpy()

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(max_iter=10000, class_weight="balanced")),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X, y)

    coef = clf.named_steps["svm"].coef_

    # średnia ważność po klasach
    importance = np.mean(np.abs(coef), axis=0)

    return pd.Series(importance, index=feature_cols).sort_values(ascending=False)
