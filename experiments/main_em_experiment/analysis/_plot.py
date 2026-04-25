"""Plot primitives: a single 'EM-rate heatmap with (coherent%) annotation' helper,
and a 'paired bars (EM, coherent)' helper, both used by both result PDFs."""

import matplotlib.pyplot as plt
import numpy as np


def heatmap_em_with_coh(ax, em_mat, coh_mat, *, row_labels, col_labels, title,
                        vmin=0, vmax=100, cmap="Reds", text_fmt="{em:.0f} ({coh:.0f})"):
    """One heatmap: cell color = EM rate; cell text = "EM (coh)"."""
    em_mat = np.asarray(em_mat, dtype=float)
    coh_mat = np.asarray(coh_mat, dtype=float)
    im = ax.imshow(em_mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=11)

    for i in range(em_mat.shape[0]):
        for j in range(em_mat.shape[1]):
            em_v = em_mat[i, j]
            coh_v = coh_mat[i, j]
            if np.isnan(em_v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
                continue
            color = "white" if em_v > 60 else "black"
            ax.text(
                j, i,
                text_fmt.format(em=em_v, coh=coh_v if not np.isnan(coh_v) else 0),
                ha="center", va="center", fontsize=7, color=color,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="EM rate %")


def bars_em_coh(ax, labels, em_vals, coh_vals, *, title):
    """Side-by-side bars per label: EM rate (red) and coherent rate (blue)."""
    em_vals = np.asarray(em_vals, dtype=float)
    coh_vals = np.asarray(coh_vals, dtype=float)
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w / 2, em_vals, w, label="EM rate %", color="#d62728")
    ax.bar(x + w / 2, coh_vals, w, label="Coherent rate %", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    for xi, em_v, coh_v in zip(x, em_vals, coh_vals):
        if not np.isnan(em_v):
            ax.text(xi - w / 2, em_v + 1, f"{em_v:.0f}", ha="center", fontsize=7)
        if not np.isnan(coh_v):
            ax.text(xi + w / 2, coh_v + 1, f"{coh_v:.0f}", ha="center", fontsize=7)
