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

nlp = spacy.load("pl_core_news_sm")

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


def evaluate_feature_set(df_feats, feature_cols, n_splits=5):
    """
    SVM (LinearSVC) na zadanym zestawie cech.
    Zwraca tablice f1 i acc dla każdego foldu.
    """
    X = df_feats[feature_cols].values
    y = df_feats["author"].values
    groups = df_feats["title"].values

    gkf = GroupKFold(n_splits=n_splits)

    f1_list, acc_list = [], []

    splits = list(gkf.split(X, y, groups))

    for train_idx, test_idx in splits:
        clf = Pipeline(
            [("scaler", StandardScaler()), ("svm", LinearSVC(max_iter=5000))]
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])

        f1_list.append(f1_score(y[test_idx], pred, average="macro"))
        acc_list.append(accuracy_score(y[test_idx], pred))

    return np.array(f1_list), np.array(acc_list), splits


def get_feature_importance(df_feats, feature_cols):
    """Trenuje SVM na całości i zwraca |wagi| per cecha."""
    X = df_feats[feature_cols].values
    y = df_feats["author"].values

    clf = Pipeline([("scaler", StandardScaler()), ("svm", LinearSVC(max_iter=5000))])
    clf.fit(X, y)

    importance = np.max(np.abs(clf.named_steps["svm"].coef_), axis=0)
    return pd.Series(importance, index=feature_cols).sort_values(ascending=False)
