# plots.py

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ============================================================
# STYL
# ============================================================

PALETTE = {
    "no_genre": "#355070",
    "genre": "#E56B6F",
    "neutral": "#6D597A",
    "grid": "#E9ECEF",
    "text": "#22223B",
    "bg": "#F8F9FA",
}

RANDOM_STATE = 42

AUTHORS_SHORT = {
    "Adam_Mickiewicz": "Mickiewicz",
    "Juliusz_Słowacki": "Słowacki",
    "Zygmunt_Krasiński": "Krasiński",
}

# ============================================================
# POMOCNICZE
# ============================================================


def style_ax(ax):

    ax.set_facecolor(PALETTE["bg"])

    ax.grid(
        axis="y",
        color=PALETTE["grid"],
        linewidth=0.8,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        labelsize=10,
    )


# ============================================================
# GŁÓWNY WYKRES
# ============================================================


def plot_results(
    result: dict,
    df_chunks: pd.DataFrame,
    save_path: str = "wyniki_autorstwo.png",
):

    fig = plt.figure(
        figsize=(18, 14),
    )

    fig.patch.set_facecolor(PALETTE["bg"])

    gs = GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.35,
        wspace=0.25,
    )

    # ========================================================
    # AXES
    # ========================================================

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        style_ax(ax)

    # ========================================================
    # WYKRES 1 — F1
    # ========================================================

    data_f1 = [
        result["f1_no"],
        result["f1_gen"],
    ]

    labels = [
        "Bez kontroli\ngatunku",
        "Z kontrolą\ngatunku",
    ]

    bp = ax1.boxplot(
        data_f1,
        labels=labels,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(
            color="white",
            linewidth=2.2,
        ),
        whiskerprops=dict(
            linewidth=1.2,
        ),
        capprops=dict(
            linewidth=1.2,
        ),
    )

    colors = [
        PALETTE["no_genre"],
        PALETTE["genre"],
    ]

    for patch, color in zip(
        bp["boxes"],
        colors,
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    # jitter
    rng = np.random.default_rng(RANDOM_STATE)

    for i, values in enumerate(data_f1, start=1):
        jitter = rng.uniform(
            -0.05,
            0.05,
            size=len(values),
        )

        ax1.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=55,
            edgecolors="white",
            linewidths=1,
            zorder=5,
            alpha=0.9,
        )

    ax1.set_ylim(0, 1)

    ax1.set_ylabel(
        "F1-score (macro)",
        fontsize=11,
    )

    ax1.set_title(
        "Porównanie F1-score",
        fontsize=13,
        fontweight="bold",
    )

    # ========================================================
    # WYKRES 2 — G-MEAN
    # ========================================================

    data_gmean = [
        result["gmean_no"],
        result["gmean_gen"],
    ]

    bp2 = ax2.boxplot(
        data_gmean,
        labels=labels,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(
            color="white",
            linewidth=2.2,
        ),
        whiskerprops=dict(
            linewidth=1.2,
        ),
        capprops=dict(
            linewidth=1.2,
        ),
    )

    for patch, color in zip(
        bp2["boxes"],
        colors,
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    rng = np.random.default_rng(RANDOM_STATE + 1)

    for i, values in enumerate(data_gmean, start=1):
        jitter = rng.uniform(
            -0.05,
            0.05,
            size=len(values),
        )

        ax2.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=55,
            edgecolors="white",
            linewidths=1,
            zorder=5,
            alpha=0.9,
        )

    ax2.set_ylim(0, 1)

    ax2.set_ylabel(
        "Geometric Mean",
        fontsize=11,
    )

    ax2.set_title(
        "Porównanie G-Mean",
        fontsize=13,
        fontweight="bold",
    )

    # ========================================================
    # WYKRES 3 — DISTRIBUTION
    # ========================================================

    dist = df_chunks.groupby(["kind", "author"]).size().reset_index(name="n")

    kinds = sorted(dist["kind"].unique())

    authors = sorted(dist["author"].unique())

    x = np.arange(len(kinds))

    width = 0.24

    author_colors = [
        "#355070",
        "#B56576",
        "#6D597A",
    ]

    for i, (author, color) in enumerate(zip(authors, author_colors)):
        vals = []

        for kind in kinds:
            row = dist[(dist["kind"] == kind) & (dist["author"] == author)]

            vals.append(row["n"].values[0] if not row.empty else 0)

        bars = ax3.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=AUTHORS_SHORT.get(author, author),
            color=color,
            alpha=0.9,
            zorder=3,
        )

        for bar, val in zip(
            bars,
            vals,
        ):
            if val > 0:
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 1,
                    str(val),
                    ha="center",
                    fontsize=9,
                )

            else:
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    1,
                    "BRAK",
                    ha="center",
                    fontsize=8,
                    color="red",
                    fontweight="bold",
                )

    ax3.set_xticks(x)

    ax3.set_xticklabels(
        kinds,
        fontsize=11,
    )

    ax3.set_ylabel(
        "Liczba chunków",
        fontsize=11,
    )

    ax3.set_title(
        "Rozkład danych\n(author × gatunek)",
        fontsize=13,
        fontweight="bold",
    )

    ax3.legend(
        fontsize=9,
        framealpha=0.7,
    )

    # ========================================================
    # WYKRES 4 — PER GENRE
    # ========================================================

    per_genre = result["per_genre"]

    genres = [x["genre"] for x in per_genre]

    means = [x["gmean_mean"] for x in per_genre]

    stds = [x["gmean_std"] for x in per_genre]

    bars = ax4.bar(
        genres,
        means,
        yerr=stds,
        capsize=6,
        width=0.45,
        color=PALETTE["genre"],
        alpha=0.9,
        zorder=3,
    )

    for bar, val in zip(
        bars,
        means,
    ):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # baseline
    n_classes = df_chunks["author"].nunique()

    baseline = 1 / n_classes

    ax4.axhline(
        baseline,
        linestyle="--",
        linewidth=1.5,
        color=PALETTE["neutral"],
        label=f"Baseline ({baseline:.2f})",
    )

    ax4.set_ylim(0, 1)

    ax4.set_ylabel(
        "Geometric Mean",
        fontsize=11,
    )

    ax4.set_title(
        "G-Mean per gatunek",
        fontsize=13,
        fontweight="bold",
    )

    ax4.legend(
        fontsize=9,
        framealpha=0.7,
    )

    # ========================================================
    # TYTUŁ
    # ========================================================

    fig.suptitle(
        ("Wpływ kontroli gatunku na klasyfikację autora"),
        fontsize=18,
        fontweight="bold",
        color=PALETTE["text"],
        y=0.98,
    )

    # ========================================================
    # ZAPIS
    # ========================================================

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.show()

    print(f"✓ Wykres zapisany: {save_path}")
