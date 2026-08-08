"""
BSM-RLI Comparative Results Plotter
Reads baseline_sweep.json + finetuned_sweep.json and generates
a delta comparison chart for all evaluated models.
Run after both sweeps are complete.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASELINE_PATH = "benchmarks/results/baseline_sweep.json"
FINETUNED_PATH = "benchmarks/results/finetuned_sweep.json"
OUT_PATH = "experiments/plots/multi_model_sweep_comparison.png"


def plot_comparison():
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(FINETUNED_PATH) as f:
        finetuned = json.load(f)

    keys, base_acc, ft_acc, deltas = [], [], [], []

    for key in baseline:
        b = baseline[key].get("accuracy_pct")
        ft = finetuned.get(key, {}).get("accuracy_pct")
        if b is not None and ft is not None:
            keys.append(key.replace("-4bit", "").replace("-bnb", ""))
            base_acc.append(b)
            ft_acc.append(ft)
            deltas.append(round(ft - b, 1))

    x = np.arange(len(keys))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), dpi=200)

    # Top chart: Baseline vs Fine-tuned accuracy
    rects1 = ax1.bar(x - width / 2, base_acc, width, label="Baseline (No FT)", color="#ff6b6b")
    rects2 = ax1.bar(x + width / 2, ft_acc, width, label="BSM-RLI Fine-Tuned (150 steps)", color="#4cc9f0")

    ax1.set_ylabel("GSM8K Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("BSM-RLI Multi-Model Sweep: Baseline vs Fine-Tuned Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys, rotation=30, ha="right", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(frameon=True, facecolor="#1e1e2e")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    # Bottom chart: Delta gain
    delta_colors = ["#2a9d8f" if d >= 0 else "#e76f51" for d in deltas]
    bars = ax2.bar(x, deltas, color=delta_colors, width=0.6)
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_ylabel("ΔS Accuracy Gain (pp)", fontsize=11, fontweight="bold")
    ax2.set_title("Absolute Accuracy Delta After 150-Step BSM-RLI Fine-Tuning (ΔS)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(keys, rotation=30, ha="right", fontsize=9, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, d in zip(bars, deltas):
        h = bar.get_height()
        ax2.annotate(f"+{d}pp" if d >= 0 else f"{d}pp",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4 if h >= 0 else -12),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout(pad=3.0)
    os.makedirs("experiments/plots", exist_ok=True)
    plt.savefig(OUT_PATH)
    plt.close()
    print(f"Saved comparison plot to {OUT_PATH}")


if __name__ == "__main__":
    if not os.path.exists(BASELINE_PATH):
        print(f"Baseline results not found at {BASELINE_PATH}. Run baseline_sweep.py first.")
    elif not os.path.exists(FINETUNED_PATH):
        print(f"Fine-tuned results not found at {FINETUNED_PATH}. Run finetuned_sweep.py first.")
    else:
        plot_comparison()
