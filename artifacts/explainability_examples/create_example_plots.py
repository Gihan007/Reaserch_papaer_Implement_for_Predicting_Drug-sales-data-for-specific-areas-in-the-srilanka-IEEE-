from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)


def summary_plot():
    features = ["sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_4", "sales_lag_5"]
    strengths = [5.4, 4.1, 3.1, 2.3, 1.5]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for row, (feature, strength) in enumerate(zip(features, strengths)):
        values = rng.uniform(0, 1, 55)
        effects = strength * (2 * values - 1) + rng.normal(0, strength * 0.22, len(values))
        jitter = rng.normal(0, 0.09, len(values))
        sc = ax.scatter(effects, row + jitter, c=values, cmap="coolwarm", vmin=0, vmax=1,
                        s=38, alpha=0.85, edgecolors="none")
    ax.axvline(0, color="#555", lw=1)
    ax.set_yticks(range(len(features)), features)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value: effect on predicted weekly sales")
    ax.set_title("Illustrative SHAP summary plot — many predictions", weight="bold")
    ax.grid(axis="x", alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Historical sales value")
    cbar.set_ticks([0, 1], labels=["Low", "High"])
    fig.text(0.5, 0.01, "Each dot = one historical prediction. Left lowers the forecast; right raises it.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "shap_summary_example.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def waterfall_plot():
    base = 35.0
    names = ["sales_lag_1 = 50", "sales_lag_2 = 45", "sales_lag_3 = 35",
             "sales_lag_4 = 40", "sales_lag_5 = 30"]
    contributions = [7.0, 4.0, 1.0, 2.0, -1.0]
    starts = [base]
    for value in contributions[:-1]:
        starts.append(starts[-1] + value)
    final = base + sum(contributions)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(names))
    for i, (name, contribution, start) in enumerate(zip(names, contributions, starts)):
        color = "#e74c3c" if contribution > 0 else "#2471a3"
        left = min(start, start + contribution)
        ax.barh(i, abs(contribution), left=left, height=0.56, color=color)
        sign = "+" if contribution > 0 else ""
        ax.text(start + contribution / 2, i, f"{sign}{contribution:.0f}", color="white",
                ha="center", va="center", weight="bold", fontsize=11)
        if i < len(names) - 1:
            ax.plot([start + contribution, start + contribution], [i + 0.28, i + 0.72],
                    color="#777", ls="--", lw=1)
    ax.axvline(base, color="#555", ls="--", lw=1.5, label=f"Base value = {base:.0f}")
    ax.axvline(final, color="#111", lw=2, label=f"Final prediction = {final:.0f}")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted weekly sales")
    ax.set_title("Illustrative SHAP waterfall plot — one prediction", weight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.18)
    fig.text(0.5, 0.01, "Red pushes the prediction higher; blue pushes it lower. 35 + 7 + 4 + 1 + 2 − 1 = 48.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "shap_waterfall_example.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


summary_plot()
waterfall_plot()
