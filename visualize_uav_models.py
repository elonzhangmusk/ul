from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path('/home/zhanghangning/ultralytics/runs/visualize')
OUTPUT_PATH = OUTPUT_DIR / 'uav_models_single_figure.png'


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            ['YOLO12 (baseline)', 0.7628, 0.9507, 2.57, 6.48, 9.29, 107.63],
            ['YOLO12 + NWD', 0.7638, 0.9525, 2.57, 6.48, 9.13, 109.51],
            ['YOLO12 + Ghost', 0.7520, 0.9515, 2.03, 6.05, 7.89, 126.74],
            ['YOLO12 + Ghost (only)', 0.7102, 0.9413, 2.03, 6.05, 8.49, 117.85],
            ['YOLO12 + NWD + Ghost (Ours)', 0.7596, 0.9518, 2.03, 6.05, 8.02, 124.69],
        ],
        columns=['Model', 'mAP', 'AP50', 'Params', 'GFLOPs', 'Latency', 'FPS'],
    )

    metric_cols = ['mAP', 'AP50', 'Params', 'GFLOPs', 'Latency', 'FPS']
    higher_better = {'mAP': True, 'AP50': True, 'Params': False, 'GFLOPs': False, 'Latency': False, 'FPS': True}

    norm = pd.DataFrame(index=df.index)
    for col in metric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if np.isclose(col_max, col_min):
            norm[col] = 1.0
        elif higher_better[col]:
            norm[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            norm[col] = (col_max - df[col]) / (col_max - col_min)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 5.8), dpi=240)
    heat = ax.imshow(norm[metric_cols].values, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_xticklabels(['mAP(0.5:0.95)', 'AP50', 'Params(M)', 'GFLOPs', 'Latency(ms)', 'FPS'], rotation=15)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['Model'])
    ax.set_title('Overall Comparison of YOLO12 Variants on EVD4UAV', fontsize=16, weight='bold', pad=14)

    for i in range(len(df)):
        for j, col in enumerate(metric_cols):
            raw = df.iloc[i][col]
            text = f'{raw:.4f}' if col in {'mAP', 'AP50'} else f'{raw:.2f}'
            ax.text(j, i, text, ha='center', va='center', color='black', fontsize=9, fontweight='bold')

    cbar = fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Normalized score (higher is better)', rotation=90)

    fig.text(
        0.5,
        0.02,
        'Green-yellow cells indicate better overall values after accounting for whether each metric should be maximized or minimized.',
        ha='center',
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUTPUT_PATH, bbox_inches='tight')
    print(OUTPUT_PATH)


if __name__ == '__main__':
    main()
