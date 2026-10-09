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
        ax.axvspan(0.84, 1.15, color=GREY, alpha=0.15, lw=0)  # rough range of European banks, see README
        ax.set_title(title)
        ax.set_xlabel("mean threshold tau")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper right")
    top = axes[2].get_ylim()[1]
    axes[2].text(0.995, 0.62 * top, "rough range of\nEuropean banks\n(WP 545 averages)", color=MUTED,
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
    ax.set_xlabel("average number of borrowers per bank")
    ax.set_ylabel("share")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robust yet fragile: above about 4 borrowers per bank a crisis\nis rarer, but when it comes it takes nearly every bank")
    fig.tight_layout()
    fig.savefig("figures/gai_kapadia.png", dpi=160)


ORDER = ["no model: count of single killers", "boosting rung 1", "boosting rung 2", "boosting rung 3", "mlp rung 3",
         "gnn (rung 3 inputs)", "post-hoc: gnn, early stopping", "boosting rung 4", "mlp rung 4", "boosting rung 5",
         "no model: simulator, 2 rounds"]
NICE = {"no model: count of single killers": "no model: one hand count",
        "boosting rung 1": "boosting: own threshold", "boosting rung 2": "boosting: + size, degree",
        "boosting rung 3": "boosting: + own exposures", "mlp rung 3": "MLP: own features",
        "gnn (rung 3 inputs)": "GNN: own features + links",
        "post-hoc: gnn, early stopping": "GNN with early stopping (added later)",
        "boosting rung 4": "boosting: + neighbour features",
        "mlp rung 4": "MLP: + neighbour features",
        "boosting rung 5": "boosting: + centrality",
        "no model: simulator, 2 rounds": "reference: simulator cut at 2 rounds"}


def ladder():
    df = pd.read_csv("results/scores.csv")
    df = df[(df.label == "y") & (df.train == df.test)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, sharex=True)
    for ax, (fam, (name, color)) in zip(axes, FAMILY.items()):
        per_seed = df[df.train == fam].groupby(["model", "seed"]).concordance.mean()
        d = per_seed.groupby("model").agg(["mean", "min", "max"]).reindex(ORDER)
        y = range(len(ORDER))[::-1]
        ax.hlines(y, d["min"], d["max"], color=color, lw=2)
        ax.scatter(d["mean"], y, color=color, s=45, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        for yi, m in zip(y, d["mean"]):
            ax.text(m + 0.012, yi, f"{m:.2f}", ha="left", va="center", fontsize=8, color=INK)
        ax.set_yticks(list(y), [NICE[m] for m in ORDER])
        ax.set_title(name)
        ax.set_xlabel("pairs of banks put in the right order (0.5 = chance)")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(ORDER) - 0.4)
        ax.set_xlim(0.65, 1.0)
    fig.suptitle("Who finds the fragile banks? Dot = mean over test networks and seeds, line = range over 10 seeds",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("figures/ladder.png", dpi=160)


def control():
    df = pd.read_csv("results/control.csv")
    series = [("simulator (reference)", "simulator on the new links (reference)", GREY, "--"),
              ("boosting rung 3", "boosting: own exposures", YELLOW, "-"),
              ("boosting rung 4", "boosting: + neighbour features", AQUA, "-"),
              ("gnn (rung 3 inputs)", "GNN", BLUE, "-")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, sharex=True)
    for ax, (fam, (name, _)) in zip(axes, FAMILY.items()):
        g = df[df.family == fam].groupby(["model", "fraction"])[["links_moved", "vs_old_labels"]].mean()
        for key, label, color, style in series:
            d = g.loc[key]
            ax.plot(d.links_moved * 100, d.vs_old_labels, marker="o", color=color, ls=style, label=label)
        ax.set_title(name)
        ax.set_xlabel("links that changed place (%)")
        ax.set_ylim(0.6, 1.02)
        ax.set_xlim(-3, 100)
    axes[0].set_ylabel("rank correlation with the ORIGINAL labels")
    axes[0].legend(loc="lower left", fontsize=8, ncols=1)
    fig.suptitle("Control: move the links, keep the old labels. Models that use the links fall; the others do not move",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("figures/control.png", dpi=160)


if __name__ == "__main__":
    regimes()
    gai_kapadia()
    ladder()
    control()
