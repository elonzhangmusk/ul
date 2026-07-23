from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("/home/zhanghangning/ultralytics/runs/visualize")
OUTPUT_PATH = OUTPUT_DIR / "uav_models_nature_style_v2.png"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            ["YOLO12 (baseline)", 0.7628, 0.9507, 2.57, 6.48, 9.29, 107.63],
            ["YOLO12 + NWD", 0.7638, 0.9525, 2.57, 6.48, 9.13, 109.51],
            ["YOLO12 + Ghost", 0.7102, 0.9413, 2.03, 6.05, 8.49, 117.85],
            ["YOLO12 + NWD + Ghost (Ours)", 0.7520, 0.9515, 2.03, 6.05, 7.89, 126.74],
        ],
        columns=["Model", "mAP", "AP50", "Params", "GFLOPs", "Latency", "FPS"],
    )

    palette = {
        "YOLO12 (baseline)": "#9A9A9A",
        "YOLO12 + NWD": "#4C78A8",
        "YOLO12 + Ghost": "#59A14F",
        "YOLO12 + NWD + Ghost (Ours)": "#E15759",
    }

    short_names = {
        "YOLO12 (baseline)": "Baseline",
        "YOLO12 + NWD": "NWD",
        "YOLO12 + Ghost": "Ghost",
        "YOLO12 + NWD + Ghost (Ours)": "Ours",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
        }
    )

    fig = plt.figure(figsize=(10.2, 4.6), dpi=320)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    for _, row in df.iterrows():
        size = 1150 / row["Params"]
        edge = "#111111" if "Ours" in row["Model"] else "white"
        lw = 1.6 if "Ours" in row["Model"] else 0.9
        ax1.scatter(
            row["Latency"],
            row["mAP"],
            s=size,
            color=palette[row["Model"]],
            alpha=0.97,
            edgecolor=edge,
            linewidth=lw,
            zorder=3,
        )

    offsets = {
        "YOLO12 (baseline)": (0.03, 0.0010),
        "YOLO12 + NWD": (0.03, 0.0022),
        "YOLO12 + Ghost": (0.03, -0.0042),
        "YOLO12 + NWD + Ghost (Ours)": (0.03, 0.0015),
    }
    for _, row in df.iterrows():
        dx, dy = offsets[row["Model"]]
        ax1.text(
            row["Latency"] + dx,
            row["mAP"] + dy,
            short_names[row["Model"]],
            fontsize=10,
            color="#111111",
            ha="left",
            va="center",
            zorder=4,
        )

    ax1.set_xlabel("Latency (ms)", labelpad=8)
    ax1.set_ylabel("mAP (0.5:0.95)", labelpad=8)
    ax1.set_title("Accuracy-efficiency trade-off", loc="left", fontsize=13, weight="bold", pad=8)
    ax1.set_xlim(7.72, 9.42)
    ax1.set_ylim(0.704, 0.768)
    ax1.grid(axis="both", color="#D8D8D8", linewidth=0.8, alpha=0.75)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(-0.12, 1.03, "a", transform=ax1.transAxes, fontsize=14, fontweight="bold")

    # Bubble legend moved away from axes and labels
    legend_x = [8.96, 9.19]
    legend_y = 0.7087
    for params, xpos in zip([2.57, 2.03], legend_x):
        ax1.scatter(xpos, legend_y, s=1150 / params, color="none", edgecolor="#666666", linewidth=1.0, zorder=2)
        ax1.text(xpos, legend_y - 0.0042, f"{params:.2f}M", ha="center", va="top", fontsize=8.8, color="#444444")
    ax1.text(sum(legend_x) / 2, legend_y + 0.0038, "Bubble size: Params", ha="center", fontsize=8.8, color="#444444")

    ax2 = fig.add_subplot(gs[0, 1])
    rank_df = df.sort_values("FPS", ascending=True)
    bars = ax2.barh(
        [short_names[m] for m in rank_df["Model"]],
        rank_df["FPS"],
        color=[palette[m] for m in rank_df["Model"]],
        height=0.56,
        alpha=0.95,
    )
    ax2.set_xlabel("FPS", labelpad=8)
    ax2.set_title("Speed ranking", loc="left", fontsize=13, weight="bold", pad=8)
    ax2.grid(axis="x", color="#D8D8D8", linewidth=0.8, alpha=0.75)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.text(-0.18, 1.03, "b", transform=ax2.transAxes, fontsize=14, fontweight="bold")
    ax2.set_xlim(100, 132)

    for bar, (_, row) in zip(bars, rank_df.iterrows()):
        label = f"{row['FPS']:.2f} | AP50 {row['AP50']:.4f}"
        ax2.text(
            row["FPS"] + 0.6, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8.8, color="#222222"
        )

    fig.suptitle("Comparison of YOLO12 variants on EVD4UAV", y=1.02, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.02,
        "NWD achieves the best detection accuracy, while the combined model provides the strongest speed-efficiency trade-off.",
        ha="center",
        fontsize=9.8,
        color="#333333",
    )

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
