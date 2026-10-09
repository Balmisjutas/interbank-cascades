"""Draw the figures of the README from the CSV files in results/."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a63", "#e4e3dc"
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8980"
FAMILY = {"er": ("random network", BLUE), "cp": ("core-periphery network", ORANGE)}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
})


def regimes():
    df = pd.read_csv("results/regimes.csv").groupby(["family", "tau"]).mean(numeric_only=True).reset_index()
    panels = [("share_contagion", "Share of failures that\nbring down another bank"),
              ("mean_cascade", "Banks failed by contagion,\naverage (of 100)"),
              ("worst_cascade", "Banks failed by contagion,\nworst case (of 100)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    for ax, (col, title) in zip(axes, panels):
        for fam, (name, color) in FAMILY.items():
            d = df[df.family == fam]
            ax.plot(d.tau, d[col], marker="o", color=color, label=name)
        ax.axvspan(0.85, 1.1, color=GREY, alpha=0.15, lw=0)
        ax.set_title(title)
        ax.set_xlabel("mean threshold tau")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper right")
    top = axes[2].get_ylim()[1]
    axes[2].text(0.975, 0.62 * top, "rough range of\nEuropean banks\n(WP 545 averages)", color=MUTED,
                 fontsize=8, ha="center", va="center")
    fig.suptitle("Contagion needs weak banks: at tau = 0.4 or below, the worst case takes most of the system",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("figures/regimes.png", dpi=160)


def gai_kapadia():
    df = pd.read_csv("results/gai_kapadia.csv").groupby("z").mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(df.z, df.frequency, marker="o", color=BLUE)
    ax.plot(df.z, df.extent, marker="o", color=ORANGE)
    ax.text(6.5, 0.50, "how often a failure\nbecomes a crisis", color=INK, fontsize=9, ha="right", va="top")
    ax.text(7.2, 0.93, "share of banks that fail\nin a crisis", color=INK, fontsize=9, ha="left", va="top")
    ax.set_xlabel("average number of counterparties per bank")
    ax.set_ylabel("share")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robust yet fragile: with more links a crisis is rarer,\nbut when it comes it takes every bank")
    fig.tight_layout()
    fig.savefig("figures/gai_kapadia.png", dpi=160)


ORDER = ["boosting rung 1", "boosting rung 2", "boosting rung 3", "mlp rung 3", "gnn (rung 3 inputs)",
         "boosting rung 4", "mlp rung 4", "boosting rung 5"]
NICE = {"boosting rung 1": "boosting: own threshold", "boosting rung 2": "boosting: + size, degree",
        "boosting rung 3": "boosting: + own exposures", "mlp rung 3": "MLP: own features",
        "gnn (rung 3 inputs)": "GNN: own features + links",
        "boosting rung 4": "boosting: + neighbour features", "mlp rung 4": "MLP: + neighbour features",
        "boosting rung 5": "boosting: + centrality"}


def ladder():
    df = pd.read_csv("results/scores.csv")
    df = df[(df.label == "y") & (df.train == df.test)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, sharex=True)
    for ax, (fam, (name, color)) in zip(axes, FAMILY.items()):
        d = df[df.train == fam].groupby("model").spearman.agg(["mean", "min", "max"]).reindex(ORDER)
        y = range(len(ORDER))[::-1]
        ax.hlines(y, d["min"], d["max"], color=color, lw=2)
        ax.scatter(d["mean"], y, color=color, s=45, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        for yi, m in zip(y, d["mean"]):
            ax.text(m, yi + 0.28, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_yticks(list(y), [NICE[m] for m in ORDER])
        ax.set_title(name)
        ax.set_xlabel("rank correlation with the true fragility (test networks)")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(ORDER) - 0.2)
    fig.suptitle("Who finds the fragile banks? Dot = mean over 10 seeds, line = min to max",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("figures/ladder.png", dpi=160)


def control():
    df = pd.read_csv("results/control.csv")
    series = [("simulator (reference)", "simulator on the new links (reference)", GREY, "--"),
              ("boosting rung 3", "boosting: own exposures", YELLOW, "-"),
              ("boosting rung 4", "boosting: + neighbour features", AQUA, "-"),
              ("gnn (rung 3 inputs)", "GNN", BLUE, "-")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (fam, (name, _)) in zip(axes, FAMILY.items()):
        d = df[df.family == fam].groupby(["model", "fraction"]).vs_old_labels.mean().unstack(0)
        for key, label, color, style in series:
            ax.plot(d.index * 100, d[key], marker="o", color=color, ls=style, label=label)
        ax.set_title(name)
        ax.set_xlabel("links moved at random (%), each bank keeps its number of links")
        ax.set_ylim(0.6, 1.02)
    axes[0].set_ylabel("rank correlation with the ORIGINAL labels")
    axes[0].legend(loc="lower left", fontsize=8, ncols=2)
    fig.suptitle("Control: move the links, keep the old labels. A model that reads the links must fall with the reference",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("figures/control.png", dpi=160)


if __name__ == "__main__":
    regimes()
    gai_kapadia()
    ladder()
    control()
