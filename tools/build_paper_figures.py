#!/usr/bin/env python3
"""Build figures for the AAAI 2027 anonymous submission."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path("/opt/tiger/hand/paper/submission_ready/Figures")


def rounded(ax, xy, wh, fc, ec, lw=1.5, r=0.03, pad=0.01):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    return patch


def token(ax, xy, wh, text, fc, ec, fontsize=11.0, weight="bold", color="#222222"):
    rounded(ax, xy, wh, fc=fc, ec=ec, lw=1.0, r=0.02, pad=0.004)
    x, y = xy
    w, h = wh
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color=color)


def build_overview():
    fig, ax = plt.subplots(figsize=(12.6, 4.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bg = Rectangle((0, 0), 1, 1, facecolor="#fcfaf6", edgecolor="none")
    ax.add_patch(bg)
    ax.plot([0.315, 0.315], [0.12, 0.87], color="#ddd4c7", lw=1.0)
    ax.plot([0.735, 0.735], [0.12, 0.87], color="#ddd4c7", lw=1.0)

    ax.text(
        0.02,
        0.96,
        "Temporal Hand Labanotation",
        fontsize=18.0,
        fontweight="bold",
        va="top",
        color="#1d1d1b",
    )
    ax.text(
        0.02,
        0.905,
        "Old HL stores framewise states. Temporal HL stores a segment object (G, M) and a deterministic repair map.",
        fontsize=10.9,
        va="top",
        color="#444444",
    )

    # Left panel
    ax.text(0.03, 0.82, "Old HL", fontsize=13.8, fontweight="bold", color="#8f3f30")
    ax.text(0.03, 0.782, "framewise flat state vector x_t", fontsize=10.0, color="#7a4c41")
    rounded(ax, (0.042, 0.29), (0.225, 0.40), fc="#f8e9e4", ec="#b86b58", lw=1.8, r=0.03)
    ax.text(0.064, 0.655, "x_t = [s_t^1, ..., s_t^R]", fontsize=11.0, fontweight="bold", color="#6e372b")
    y_rows = [0.58, 0.49, 0.40]
    x_cols = [0.066, 0.132, 0.198]
    for y in y_rows:
        for x in x_cols:
            token(ax, (x, y), (0.043, 0.043), "S", "#fff8f5", "#d59d90", fontsize=11.2, color="#5f524d")
    ax.text(0.154, 0.338, "flat regional symbols,", ha="center", fontsize=10.0, color="#6f4035")
    ax.text(0.154, 0.306, "no explicit transition object", ha="center", fontsize=10.0, color="#6f4035")

    # Middle panel
    ax.text(0.36, 0.82, "Temporal HL", fontsize=13.8, fontweight="bold", color="#245e48")
    ax.text(0.36, 0.782, "segment object (G, M) + repair operator", fontsize=10.0, color="#45685a")
    rounded(ax, (0.352, 0.27), (0.335, 0.45), fc="#e9f4ee", ec="#45886d", lw=1.9, r=0.034)
    ax.text(0.519, 0.673, "(G_{t:t+\\ell-1}, M_{t:t+\\ell-1})", ha="center", fontsize=12.0, fontweight="bold", color="#245e48")
    ax.text(0.519, 0.642, "temporal representation object", ha="center", fontsize=10.5, color="#2f5446")

    token(ax, (0.386, 0.585), (0.05, 0.045), "S", "#f8fcfa", "#89b6a0", fontsize=11.4, color="#395c4f")
    token(ax, (0.448, 0.585), (0.05, 0.045), "S", "#f8fcfa", "#89b6a0", fontsize=11.4, color="#395c4f")
    token(ax, (0.510, 0.585), (0.05, 0.045), "S", "#f8fcfa", "#89b6a0", fontsize=11.4, color="#395c4f")
    token(ax, (0.572, 0.585), (0.078, 0.045), "Δ motif", "#eff8f3", "#73a58a", fontsize=10.0, color="#2f5a49")

    rounded(ax, (0.392, 0.485), (0.235, 0.06), fc="#f7fcf8", ec="#95bea9", lw=1.0, r=0.02)
    ax.text(0.509, 0.515, "G_t: anatomy-aware grouped states", ha="center", va="center", fontsize=10.4, color="#2f5446")
    rounded(ax, (0.392, 0.405), (0.235, 0.06), fc="#f7fcf8", ec="#95bea9", lw=1.0, r=0.02)
    ax.text(0.509, 0.435, "M_{t:t+\\ell-1}: short transition motifs", ha="center", va="center", fontsize=10.0, color="#2f5446")
    rounded(ax, (0.412, 0.315), (0.195, 0.055), fc="#fbf7fd", ec="#ad94c6", lw=1.0, r=0.02)
    ax.text(0.510, 0.343, "rho(G, M): deterministic family repair", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#65527b")

    group_circle = Circle((0.405, 0.69), 0.045, facecolor="#f6ead8", edgecolor="#7a6d5d", linewidth=1.0)
    motif_circle = Circle((0.632, 0.69), 0.045, facecolor="#e4ebf5", edgecolor="#6d7480", linewidth=1.0)
    ax.add_patch(group_circle)
    ax.add_patch(motif_circle)
    ax.text(0.405, 0.69, "G", ha="center", va="center", fontsize=11.3, fontweight="bold", color="#5b5145")
    ax.text(0.632, 0.69, "M", ha="center", va="center", fontsize=11.3, fontweight="bold", color="#4f5967")
    ax.add_patch(FancyArrowPatch((0.438, 0.672), (0.467, 0.63), arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#696969"))
    ax.add_patch(FancyArrowPatch((0.600, 0.672), (0.560, 0.63), arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#696969"))

    ax.add_patch(FancyArrowPatch((0.281, 0.49), (0.336, 0.49), arrowstyle="-|>", mutation_scale=15, lw=1.7, color="#555555"))
    ax.text(0.292, 0.53, "upgrade", fontsize=10.0, color="#565656")

    # Right panel
    ax.text(0.765, 0.82, "Representation gains", fontsize=13.8, fontweight="bold", color="#224e76")
    ax.text(0.765, 0.782, "evidence focuses on representation behavior", fontsize=10.0, color="#4b5f72")

    # Structure badge
    rounded(ax, (0.76, 0.64), (0.205, 0.125), fc="#e6edf7", ec="#707070", lw=1.1, r=0.024, pad=0.012)
    ax.text(0.777, 0.735, "Structure", fontsize=11.5, fontweight="bold", color="#1d1d1d")
    xs = [0.800, 0.866, 0.934]
    for x, lab, val, size in zip(
        xs,
        ["flat", "grouped", "delta"],
        ["0.692", "0.897", "+0.205"],
        [10.9, 10.9, 9.8],
    ):
        ax.text(x, 0.694, lab, fontsize=8.8, color="#535353", ha="center")
        ax.text(x, 0.662, val, fontsize=size, fontweight="bold", color="#5b7fa7", ha="center")
    ax.text(0.833, 0.662, "·", fontsize=16.0, color="#666666", va="center")
    ax.text(0.900, 0.662, "·", fontsize=16.0, color="#666666", va="center")

    # Locality badge
    rounded(ax, (0.76, 0.465), (0.205, 0.125), fc="#ebf4ec", ec="#707070", lw=1.1, r=0.024, pad=0.012)
    ax.text(0.777, 0.56, "Locality", fontsize=11.5, fontweight="bold", color="#1d1d1d")
    ax.text(0.777, 0.518, "symbolic", fontsize=9.4, color="#545454")
    ax.text(0.777, 0.486, "0.355", fontsize=13.0, fontweight="bold", color="#5f8d6f")
    ax.text(0.858, 0.501, "vs", fontsize=11.2, color="#444444")
    ax.text(0.938, 0.518, "proxy", fontsize=9.4, color="#545454", ha="right")
    ax.text(0.938, 0.486, "0.085-0.095", fontsize=11.5, fontweight="bold", color="#5f8d6f", ha="right")

    # Temporal gain badge
    rounded(ax, (0.76, 0.29), (0.205, 0.125), fc="#f5eddc", ec="#707070", lw=1.1, r=0.024, pad=0.012)
    ax.text(0.777, 0.385, "Temporal gain", fontsize=11.5, fontweight="bold", color="#1d1d1d")
    ax.text(0.777, 0.343, "hard slice", fontsize=9.2, color="#545454")
    ax.text(0.850, 0.343, "0.025 to 0.181", fontsize=11.2, fontweight="bold", color="#a4834f")
    ax.text(0.777, 0.304, "chunk", fontsize=9.2, color="#545454")
    ax.text(0.850, 0.304, "0.703 to 0.779", fontsize=11.2, fontweight="bold", color="#a4834f")

    ax.add_patch(FancyArrowPatch((0.695, 0.49), (0.748, 0.49), arrowstyle="-|>", mutation_scale=15, lw=1.7, color="#555555"))
    ax.text(0.698, 0.53, "validate", fontsize=10.0, color="#565656")
    fig.tight_layout()
    fig.savefig(ROOT / "temporal_hl_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def build_key_results():
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))

    colors = ["#d8c7a1", "#8fb996", "#4d7ea8", "#c05746"]

    ax = axes[0, 0]
    vals = [0.6923, 0.8974, 1.0]
    ax.bar(["Flat", "Grouped", "Grouped+Family"], vals, color=colors[:3])
    ax.set_ylim(0, 1.05)
    ax.set_title("Structural Frontier")
    ax.set_ylabel("Sequence Accuracy")
    for i, label in enumerate(["0.692", "0.897", "+0.205"]):
        y = vals[i] + 0.02 if i < 2 else 1.02
        ax.text(i, y, label, ha="center", fontsize=9)

    ax = axes[0, 1]
    tasks = ["Approach", "Separate", "R-Open"]
    symbolic = [0.3553, 0.3553, 0.3471]
    proxy = [0.0945, 0.0848, 0.0722]
    x = range(len(tasks))
    ax.bar([i - 0.18 for i in x], symbolic, width=0.36, label="Symbolic", color="#8fb996")
    ax.bar([i + 0.18 for i in x], proxy, width=0.36, label="Proxy", color="#c05746")
    ax.set_xticks(list(x), tasks)
    ax.set_ylim(0, 0.42)
    ax.set_title("Weighted Proxy Contrast")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    strict = [0.0250, 0.0533]
    relax = [0.1807, 0.1812]
    x = range(2)
    ax.bar([i - 0.18 for i in x], strict, width=0.36, label="Strict", color="#d8c7a1")
    ax.bar([i + 0.18 for i in x], relax, width=0.36, label="Relax both", color="#4d7ea8")
    ax.set_xticks(list(x), ["Closing", "Opening"])
    ax.set_ylim(0, 0.22)
    ax.set_title("Hard Interaction Editing")
    ax.set_ylabel("Joint Score")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    fixed = [0.7025, 0.6306]
    chunk = [0.7785, 0.7197]
    x = range(2)
    ax.bar([i - 0.18 for i in x], fixed, width=0.36, label="Fixed", color="#d8c7a1")
    ax.bar([i + 0.18 for i in x], chunk, width=0.36, label="Chunk", color="#8fb996")
    ax.set_xticks(list(x), ["Closing", "Opening"])
    ax.set_ylim(0.55, 0.82)
    ax.set_title("Broader Feasible Chunk Transfer")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(ROOT / "temporal_hl_key_results.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    build_overview()
    build_key_results()


if __name__ == "__main__":
    main()
