import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
AUTHORS = ["Adam_Mickiewicz", "Juliusz_Słowacki", "Zygmunt_Krasiński"]
KINDS = ["Dramat", "Epika"]

PALETTE = {
    "no_genre": "#2E4057",
    "genre": "#E07A5F",
    "neutral": "#8D99AE",
    "grid": "#E8EDF2",
    "text": "#1A1A2E",
}

AUTHORS_SHORT = {
    "Adam_Mickiewicz": "Mickiewicz",
    "Juliusz_Słowacki": "Słowacki",
    "Zygmunt_Krasiński": "Krasiński",
}


def plot_all(results: list[dict], df_chunks: pd.DataFrame):
    """Generuje 4 wykresy na jednej figurze."""

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#F7F9FC")
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#F7F9FC")
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, zorder=0)

    r = results[0]  # word_unigram

    # ── Wykres 1: F1-score (boxplot) ──────────────────────────────
    bp = ax1.boxplot(
        [r["f1_no"], r["f1_gen"]],
        labels=["Bez kontroli\ngatunku", "Z kontrolą\ngatunku"],
        patch_artist=True,
        widths=0.4,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(color=PALETTE["text"], linewidth=1.2),
        capprops=dict(color=PALETTE["text"], linewidth=1.2),
        flierprops=dict(
            marker="o",
            markerfacecolor=PALETTE["neutral"],
            markersize=5,
            linestyle="none",
        ),
    )
    colors_bp = [PALETTE["no_genre"], PALETTE["genre"]]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Nakładamy jitter
    for i, data in enumerate([r["f1_no"], r["f1_gen"]], start=1):
        jitter = np.random.default_rng(RANDOM_STATE).uniform(
            -0.06, 0.06, size=len(data)
        )
        ax1.scatter(
            i + jitter,
            data,
            color=colors_bp[i - 1],
            edgecolors="white",
            s=50,
            zorder=5,
            alpha=0.9,
        )

    ax1.set_ylabel("F1-score (macro)", fontsize=11)
    ax1.set_title("Rozkład F1-score\nw obu wariantach", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1)

    # ── Wykres 2: Accuracy (boxplot) ──────────────────────────────
    bp2 = ax2.boxplot(
        [r["acc_no"], r["acc_gen"]],
        labels=["Bez kontroli\ngatunku", "Z kontrolą\ngatunku"],
        patch_artist=True,
        widths=0.4,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(color=PALETTE["text"], linewidth=1.2),
        capprops=dict(color=PALETTE["text"], linewidth=1.2),
        flierprops=dict(
            marker="o",
            markerfacecolor=PALETTE["neutral"],
            markersize=5,
            linestyle="none",
        ),
    )
    for patch, color in zip(bp2["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    for i, data in enumerate([r["acc_no"], r["acc_gen"]], start=1):
        jitter = np.random.default_rng(RANDOM_STATE + 1).uniform(
            -0.06, 0.06, size=len(data)
        )
        ax2.scatter(
            i + jitter,
            data,
            color=colors_bp[i - 1],
            edgecolors="white",
            s=50,
            zorder=5,
            alpha=0.9,
        )

    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Rozkład Accuracy\nw obu wariantach", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1)

    # ── Wykres 3: Liczba chunków per autor × gatunek ──────────────
    dist = df_chunks.groupby(["kind", "author"]).size().reset_index(name="n")
    dist["author_short"] = dist["author"].map(AUTHORS_SHORT)

    kinds = sorted(dist["kind"].unique())
    authors = [AUTHORS_SHORT[a] for a in AUTHORS]
    x = np.arange(len(kinds))
    width = 0.25
    author_colors = ["#2E4057", "#E07A5F", "#8D99AE"]

    for i, (author_full, author_short, color) in enumerate(
        zip(AUTHORS, authors, author_colors)
    ):
        vals = []
        for kind in kinds:
            row = dist[(dist["kind"] == kind) & (dist["author"] == author_full)]
            vals.append(row["n"].values[0] if not row.empty else 0)
        bars = ax3.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=author_short,
            color=color,
            alpha=0.88,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(v),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=PALETTE["text"],
                )
            else:
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    1.5,
                    "BRAK",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#CC2936",
                    fontweight="bold",
                )

    ax3.set_xticks(x)
    ax3.set_xticklabels(kinds, fontsize=11)
    ax3.set_ylabel("Liczba chunków (200 słów)", fontsize=11)
    ax3.set_title(
        "Liczba chunków per autor × gatunek\n(problemy z danymi)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.legend(fontsize=9, framealpha=0.6)

    # ── Wykres 4: F1 per gatunek (wariant z kontrolą) ────────────
    per_genre = r["per_genre"]
    genre_names = [p["genre"] for p in per_genre]
    genre_f1s = [p["f1_mean"] for p in per_genre]
    genre_stds = [p["f1_std"] for p in per_genre]
    genre_colors = [PALETTE["genre"]] * len(genre_names)

    bars4 = ax4.bar(
        genre_names, genre_f1s, color=genre_colors, alpha=0.88, width=0.4, zorder=3
    )
    ax4.errorbar(
        genre_names,
        genre_f1s,
        yerr=genre_stds,
        fmt="none",
        color=PALETTE["text"],
        capsize=5,
        linewidth=1.5,
        zorder=5,
    )

    for bar, val in zip(bars4, genre_f1s):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=PALETTE["text"],
            fontweight="bold",
        )

    # linia baseline: losowy klasyfikator
    n_classes = df_chunks["author"].nunique()
    baseline = 1 / n_classes
    ax4.axhline(
        baseline,
        linestyle="--",
        color=PALETTE["neutral"],
        linewidth=1.3,
        label=f"Baseline losowy ({baseline:.2f})",
    )
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("F1-score (macro)", fontsize=11)
    ax4.set_title(
        "F1 per gatunek\n(wariant z kontrolą gatunku)", fontsize=12, fontweight="bold"
    )
    ax4.legend(fontsize=9, framealpha=0.6)

    # ── Legenda globalna ──────────────────────────────────────────
    patch_no = mpatches.Patch(
        facecolor=PALETTE["no_genre"], label="Bez kontroli gatunku"
    )
    patch_gen = mpatches.Patch(facecolor=PALETTE["genre"], label="Z kontrolą gatunku")
    fig.legend(
        handles=[patch_no, patch_gen],
        loc="lower center",
        ncol=2,
        fontsize=11,
        framealpha=0.7,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Klasyfikacja autora\nWpływ kontroli gatunku na skuteczność",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["text"],
        y=1.01,
    )

    plt.savefig(
        "./wyniki_autorstwo.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.show()
    print("✓ Wykres zapisany: wyniki_autorstwo.png")


# ============================================================
# 8. RAPORT WYNIKOWY
# ============================================================


def print_report(results: list[dict]):
    print("\n" + "=" * 60)
    print("WYNIKI EKSPERYMENTU")
    print("=" * 60)

    for r in results:
        stats = r["stats"]
        print(f"\nWektoryzacja: {r['vectorizer']}")
        print(f"  Wariant BEZ kontroli gatunku:")
        print(
            f"    F1-macro  = {stats['f1_no_mean']:.4f}  ± {stats['f1_no_std']:.4f}"
            f"  (n_foldów={stats['n_no']})"
        )
        print(
            f"    Accuracy  = {np.mean(r['acc_no']):.4f}  ± {np.std(r['acc_no']):.4f}"
        )

        print(f"  Wariant Z kontrolą gatunku:")
        print(
            f"    F1-macro  = {stats['f1_gen_mean']:.4f}  ± {stats['f1_gen_std']:.4f}"
            f"  (n_obserwacji={stats['n_gen']})"
        )
        print(
            f"    Accuracy  = {np.mean(r['acc_gen']):.4f}  ± {np.std(r['acc_gen']):.4f}"
        )

        print(f"\n  Różnica F1 (z kontrolą − bez) = {stats['delta']:+.4f}")
        print(
            f"  Welch t-test:   t = {stats['welch_t']:.3f},  p = {stats['welch_p']:.4f}"
        )
        print(f"  Mann-Whitney U: U = {stats['mwu_u']:.1f},  p = {stats['mwu_p']:.4f}")
        sig = "TAK (p < 0.05)" if stats["significant"] else "NIE (p ≥ 0.05)"
        print(f"  Istotność statystyczna: {sig}")

        print(f"\n  F1 per gatunek (wariant z kontrolą):")
        for g in r["per_genre"]:
            authors_str = ", ".join(AUTHORS_SHORT.get(a, a) for a in g["authors"])
            print(
                f"    {g['genre']:8s}: F1 = {g['f1_mean']:.4f} ± {g['f1_std']:.4f}"
                f"  [{authors_str}]  {g['n_folds']}-fold"
            )
