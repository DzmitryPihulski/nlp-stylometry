import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    within_tfidf,
    between_tfidf,
    within_herbert,
    between_herbert,
    mat_tfidf,
    mat_herbert,
    authors,
    xlabels,
    stats_per_repr,
    p_w,
    p_b,
):

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # --- histogramy rozkładów (wiersz 1) ---
    for ax, (within, between, name) in zip(
        axes[0],
        [
            (within_tfidf, between_tfidf, "TF-IDF"),
            (within_herbert, between_herbert, "HerBERT"),
        ],
    ):
        ax.hist(
            within,
            bins=80,
            alpha=0.6,
            density=True,
            color="#4C72B0",
            label="Ten sam autor",
        )
        ax.hist(
            between,
            bins=80,
            alpha=0.6,
            density=True,
            color="#DD8452",
            label="Różni autorzy",
        )
        ax.axvline(
            np.mean(within),
            color="#4C72B0",
            linestyle="--",
            lw=2,
            label=f"μ within  = {np.mean(within):.3f}",
        )
        ax.axvline(
            np.mean(between),
            color="#DD8452",
            linestyle="--",
            lw=2,
            label=f"μ between = {np.mean(between):.3f}",
        )
        ax.set_title(f"Rozkład cosine similarity — {name}")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Gęstość")
        ax.legend(fontsize=8)

    # --- heatmapy średniego podobieństwa (wiersz 2) ---
    for ax, mat, name in zip(
        axes[1],
        [mat_tfidf, mat_herbert],
        ["TF-IDF", "HerBERT"],
    ):
        vmin = min(mat_tfidf.min(), mat_herbert.min())
        vmax = max(mat_tfidf.max(), mat_herbert.max())
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="YlOrRd")
        ax.set_xticks(range(len(authors)))
        ax.set_xticklabels(xlabels, rotation=15, ha="right")
        ax.set_yticks(range(len(authors)))
        ax.set_yticklabels(xlabels)
        for i in range(len(authors)):
            for j in range(len(authors)):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Średnie podobieństwo par autorów — {name}")

    plt.tight_layout()
    plt.savefig("q2_similarity_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ============================================================
    # G. INTERPRETACJA WYNIKÓW (automatyczny wydruk)
    # ============================================================

    print("\n" + "=" * 60)
    print("INTERPRETACJA WYNIKÓW")
    print("=" * 60)

    for _, row in stats_per_repr.iterrows():
        sig = "ISTOTNA STATYSTYCZNIE" if row["p_value"] < 0.05 else "NIEISTOTNA"
        disc = row["within_mean"] - row["between_mean"]
        print(f"""
    Reprezentacja: {row["reprezentacja"]}
    within-author:   {row["within_mean"]:.4f} ± {row["within_std"]:.4f}
    between-author:  {row["between_mean"]:.4f} ± {row["between_std"]:.4f}
    Różnica (dyskryminowalność): {disc:+.4f}
    Mann-Whitney U={row["U"]:.0f}, p={row["p_value"]:.4e}
    Effect size r={row["effect_r"]:.3f}
    → Różnica within vs between: {sig}
    """)

    print(f"Porównanie reprezentacji:")
    print(f"  within-author  TF-IDF vs HerBERT: p={p_w:.4e}")
    print(f"  between-author TF-IDF vs HerBERT: p={p_b:.4e}")
