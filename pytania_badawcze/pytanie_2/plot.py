import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PLOTS
# ============================================================


def plot_results(
    similarity_data,
    matrices,
    authors,
    xlabels,
    stats_per_repr,
):
    """
    similarity_data:
        {
            "TF-IDF": (within, between),
            ...
        }

    matrices:
        {
            "TF-IDF": matrix,
            ...
        }
    """

    repr_names = list(similarity_data.keys())

    n_repr = len(repr_names)

    fig, axes = plt.subplots(2, n_repr, figsize=(6 * n_repr, 10))

    # ========================================================
    # HISTOGRAMY
    # ========================================================

    for idx, name in enumerate(repr_names):
        ax = axes[0, idx]

        within, between = similarity_data[name]

        ax.hist(
            within,
            bins=80,
            alpha=0.6,
            density=True,
            label="Ten sam autor",
        )

        ax.hist(
            between,
            bins=80,
            alpha=0.6,
            density=True,
            label="Różni autorzy",
        )

        ax.axvline(
            np.mean(within),
            linestyle="--",
            lw=2,
            label=f"μ within={np.mean(within):.3f}",
        )

        ax.axvline(
            np.mean(between),
            linestyle="--",
            lw=2,
            label=f"μ between={np.mean(between):.3f}",
        )

        ax.set_title(name)

        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Gęstość")

        ax.legend(fontsize=8)

    # ========================================================
    # HEATMAPY
    # ========================================================

    global_min = min(np.nanmin(m) for m in matrices.values())

    global_max = max(np.nanmax(m) for m in matrices.values())

    for idx, name in enumerate(repr_names):
        ax = axes[1, idx]

        mat = matrices[name]

        im = ax.imshow(
            mat,
            vmin=global_min,
            vmax=global_max,
            cmap="YlOrRd",
        )

        ax.set_xticks(range(len(authors)))
        ax.set_xticklabels(
            xlabels,
            rotation=15,
            ha="right",
        )

        ax.set_yticks(range(len(authors)))
        ax.set_yticklabels(xlabels)

        for i in range(len(authors)):
            for j in range(len(authors)):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        ax.set_title(name)

        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    plt.savefig(
        "q2_similarity_analysis.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    # ========================================================
    # INTERPRETACJA
    # ========================================================

    print("\n" + "=" * 60)
    print("INTERPRETACJA WYNIKÓW")
    print("=" * 60)

    for _, row in stats_per_repr.iterrows():
        sig = "ISTOTNA" if row["p_value"] < 0.05 else "NIEISTOTNA"

        disc = row["within_mean"] - row["between_mean"]

        print(f"""
Reprezentacja: {row["reprezentacja"]}

within-author:
  {row["within_mean"]:.4f}
  ± {row["within_std"]:.4f}

between-author:
  {row["between_mean"]:.4f}
  ± {row["between_std"]:.4f}

Dyskryminowalność:
  {disc:+.4f}

Mann-Whitney:
  U={row["U"]:.0f}
  p={row["p_value"]:.4e}

Effect size:
  r={row["effect_r"]:.3f}

→ {sig}
""")
