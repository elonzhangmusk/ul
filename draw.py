import matplotlib.pyplot as plt
import numpy as np

# Baseline vs Ours
baseline = {
    "name": "Baseline",
    "AP50": 0.9507,
    "FPS": 107.63,
    "Params": 2.57,
    "GFLOPs": 6.48,
    "Latency": 9.29,
}

ours = {
    "name": "Ours",
    "AP50": 0.9515,
    "FPS": 126.74,
    "Params": 2.03,
    "GFLOPs": 6.05,
    "Latency": 7.89,
}

params_reduction = (ours["Params"] - baseline["Params"]) / baseline["Params"] * 100
gflops_reduction = (ours["GFLOPs"] - baseline["GFLOPs"]) / baseline["GFLOPs"] * 100
latency_reduction = (ours["Latency"] - baseline["Latency"]) / baseline["Latency"] * 100
fps_improvement = (ours["FPS"] - baseline["FPS"]) / baseline["FPS"] * 100

change_labels = ["Params", "GFLOPs", "Latency", "FPS"]
change_values = [
    params_reduction,
    gflops_reduction,
    latency_reduction,
    fps_improvement,
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    }
)

def make_panel_a():
    baseline_color = "#7F7F7F"
    ours_color = "#D55E00"

    fig, ax = plt.subplots(figsize=(5.3, 4.2), dpi=320)

    ax.scatter(
        baseline["FPS"],
        baseline["AP50"],
        s=190,
        color=baseline_color,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    ax.scatter(
        ours["FPS"],
        ours["AP50"],
        s=190,
        color=ours_color,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    ax.annotate(
        "",
        xy=(ours["FPS"], ours["AP50"]),
        xytext=(baseline["FPS"], baseline["AP50"]),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"),
    )

    ax.text(
        baseline["FPS"] + 0.8,
        baseline["AP50"] - 0.00045,
        "Baseline",
        fontsize=11,
        color="#222222",
        ha="left",
        va="top",
    )
    ax.text(
        ours["FPS"] - 1.0,
        ours["AP50"] + 0.00022,
        "Ours",
        fontsize=11,
        color="#222222",
        ha="right",
        va="bottom",
    )

    ax.set_xlabel("FPS", labelpad=8)
    ax.set_ylabel(r"AP$_{50}$", labelpad=8)
    ax.set_xlim(104, 129.5)
    ax.set_ylim(0.9499, 0.9519)
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("baseline_vs_ours_ap50_fps.png", bbox_inches="tight", facecolor="white")
    plt.savefig("baseline_vs_ours_ap50_fps.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_panel_b():
    neg_color = "#9CB0C3"
    pos_color = "#EAB080"
    bar_colors = [neg_color, neg_color, neg_color, pos_color]

    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=320)

    x = np.arange(len(change_labels))
    bars = ax.bar(
        x,
        change_values,
        color=bar_colors,
        width=0.58,
        edgecolor="none",
        linewidth=0.0,
        zorder=3,
    )

    ax.axhline(0, color="#D9D9D9", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(change_labels)
    ax.set_ylabel("Relative change (%)", labelpad=8)
    ax.set_ylim(-24.5, 22.5)
    ax.grid(axis="y", color="#555555", linewidth=0.8, alpha=0.8, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for rect, value in zip(bars, change_values):
        label = f"{value:+.1f}%"
        if value >= 0:
            y = value + 1.0
            va = "bottom"
        else:
            y = value - 1.0
            va = "top"
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y,
            label,
            ha="center",
            va=va,
            fontsize=11,
            color="#222222",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("baseline_vs_ours_efficiency_changes.png", bbox_inches="tight", facecolor="white")
    plt.savefig("baseline_vs_ours_efficiency_changes.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_panel_a()
    make_panel_b()
