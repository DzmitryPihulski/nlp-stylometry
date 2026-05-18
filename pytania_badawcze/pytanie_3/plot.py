import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def plot_result(
    f1_lex,
    f1_syn,
    acc_lex,
    acc_syn,
    gmean_lex,
    gmean_syn,
    t_stat_f1,
    t_stat_acc,
    t_stat_gmean,
    t_p_f1,
    t_p_acc,
    t_p_gmean,
    w_stat_f1,
    w_stat_acc,
    w_stat_gmean,
    w_p_f1,
    w_p_acc,
    w_p_gmean,
    imp_lex,
    imp_syn,
    top_n: int,
    n_splits: int,
    output_path: str = "q3_features_comparison.png",
):
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    fold_x = np.arange(1, n_splits + 1)

    # 1. F1 per fold
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fold_x, f1_lex, "o-", label="Leksykalne")
    ax1.plot(fold_x, f1_syn, "s-", label="Składniowe")
    ax1.axhline(np.mean(f1_lex), linestyle="--", alpha=0.5)
    ax1.axhline(np.mean(f1_syn), linestyle="--", alpha=0.5)
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("F1-score (macro)")
    ax1.set_title("F1-score per fold")
    ax1.set_xticks(fold_x)
    ax1.set_ylim(0, 1)
    ax1.legend()

    # 2. Boxplot wyników
    ax2 = fig.add_subplot(gs[0, 1])
    bp_data = [f1_lex, f1_syn, acc_lex, acc_syn, gmean_lex, gmean_syn]
    bp = ax2.boxplot(bp_data, patch_artist=True, widths=0.6)

    labels = ["Lex\nF1", "Syn\nF1", "Lex\nAcc", "Syn\nAcc", "Lex\nGMean", "Syn\nGMean"]
    colors = ["#4C72B0", "#DD8452", "#4C72B0", "#DD8452", "#4C72B0", "#DD8452"]

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Wartość")
    ax2.set_title("Rozkład wyników (CV)")
    ax2.set_ylim(0, 1)

    # 3. Tabela testów statystycznych
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")

    table_data = [
        ["", "F1-score", "Accuracy", "Geometric mean"],
        ["t (stat)", f"{t_stat_f1:.3f}", f"{t_stat_acc:.3f}", f"{t_stat_gmean:.3f}"],
        ["t (p-val)", f"{t_p_f1:.4f}", f"{t_p_acc:.4f}", f"{t_p_gmean:.4f}"],
        ["W (stat)", f"{w_stat_f1:.1f}", f"{w_stat_acc:.1f}", f"{w_stat_gmean:.1f}"],
        ["W (p-val)", f"{w_p_f1:.4f}", f"{w_p_acc:.4f}", f"{w_p_gmean:.4f}"],
        [
            "μ Lex",
            f"{np.mean(f1_lex):.3f}",
            f"{np.mean(acc_lex):.3f}",
            f"{np.mean(gmean_lex):.3f}",
        ],
        [
            "μ Syn",
            f"{np.mean(f1_syn):.3f}",
            f"{np.mean(acc_syn):.3f}",
            f"{np.mean(gmean_syn):.3f}",
        ],
    ]

    tbl = ax3.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.25, 1.7)
    ax3.set_title("Testy statystyczne (parowane)", pad=20)

    # 4. Top cechy leksykalne
    ax4 = fig.add_subplot(gs[1, 0:2])
    top_lex = imp_lex.head(top_n)
    labels_lex = [
        c.replace("lex_fw_", "fw: ").replace("lex_", "") for c in top_lex.index
    ]
    bars = ax4.barh(range(len(top_lex)), top_lex.values, alpha=0.8)
    ax4.set_yticks(range(len(top_lex)))
    ax4.set_yticklabels(labels_lex, fontsize=9)
    ax4.invert_yaxis()
    ax4.set_xlabel("|waga SVM|")
    ax4.set_title(f"Top {top_n} cech leksykalnych (SVM)")
    for bar, val in zip(bars, top_lex.values):
        ax4.text(
            val + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8,
        )

    # 5. Top cechy składniowe
    ax5 = fig.add_subplot(gs[1, 2])
    top_syn = imp_syn.head(top_n)
    labels_syn = [
        c.replace("syn_pos_", "POS: ").replace("syn_", "") for c in top_syn.index
    ]
    bars2 = ax5.barh(range(len(top_syn)), top_syn.values, alpha=0.8)
    ax5.set_yticks(range(len(top_syn)))
    ax5.set_yticklabels(labels_syn, fontsize=9)
    ax5.invert_yaxis()
    ax5.set_xlabel("|waga SVM|")
    ax5.set_title(f"Top {top_n} cech składniowych (SVM)")
    for bar, val in zip(bars2, top_syn.values):
        ax5.text(
            val + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8,
        )

    plt.suptitle(
        "Cechy leksykalne vs składniowe — klasyfikacja autora",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
