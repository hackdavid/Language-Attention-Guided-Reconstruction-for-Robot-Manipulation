# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Generate the result figures from the W&B summary scalars in
``evidence/metrics.md``.

Outputs (under ``evidence/figures/``):
    fig3_train_loss.png   - train action / recon / total loss bars
    fig4_per_dof_mae.png  - per-DoF validation MAE bar chart
    fig5_val_total.png    - validation total-loss comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "evidence" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----- final-step scalars copied verbatim from evidence/metrics.md -----
# (kept here as a small dict so the script is self-contained and reviewable)
RESULTS = {
    # condition: dict of metric -> value
    "C1": {
        "train_action": 0.23354, "train_recon": 0.0, "train_total": 0.23354,
        "val_total": 1.02558, "val_mae_mean": 0.80830,
        "val_mae_dim": [0.86503, 0.81300, 0.80881, 0.79185, 0.76644, 0.77084, 0.84210],
        "epoch": 2,
        "label": "C1 action-only",
    },
    "C2": {
        "train_action": 0.24451, "train_recon": 0.04637, "train_total": 0.26769,
        "val_total": 1.06714, "val_mae_mean": 0.80487,
        "val_mae_dim": [0.81118, 0.78596, 0.82105, 0.79726, 0.78653, 0.80744, 0.82469],
        "epoch": 3,
        "label": "C2 random mask",
    },
    "C3": {
        "train_action": 0.23897, "train_recon": 0.04842, "train_total": 0.26318,
        "val_total": 1.08493, "val_mae_mean": 0.80946,
        "val_mae_dim": [0.83424, 0.79289, 0.81394, 0.79587, 0.80444, 0.78470, 0.84011],
        "epoch": 3,
        "label": "C3 naive attention",
    },
    "C4": {
        "train_action": 0.24510, "train_recon": 0.05213, "train_total": 0.27117,
        "val_total": 1.08590, "val_mae_mean": 0.80925,
        "val_mae_dim": [0.83415, 0.79239, 0.81384, 0.79586, 0.80444, 0.78469, 0.83940],
        "epoch": 3,
        "label": "C4 selected heads",
    },
    "C5": {
        "train_action": 0.24510, "train_recon": 0.05228, "train_total": 0.27124,
        "val_total": 1.08413, "val_mae_mean": 0.80925,
        "val_mae_dim": [0.83415, 0.79239, 0.81384, 0.79586, 0.80444, 0.78469, 0.83940],
        "epoch": 3,
        "label": "C5 EMA teacher",
    },
}

# Distinct, colour-blind-friendly palette (Wong 2011)
COLOURS = {
    "C1": "#0072B2",  # blue
    "C2": "#E69F00",  # orange
    "C3": "#009E73",  # green
    "C4": "#D55E00",  # vermillion
    "C5": "#CC79A7",  # purple
}


def _setup(ax: plt.Axes, ylabel: str, title: str) -> None:
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def figure3_train_loss() -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    conds = list(RESULTS.keys())
    x = list(range(len(conds)))
    bar_w = 0.27
    action = [RESULTS[c]["train_action"] for c in conds]
    recon = [RESULTS[c]["train_recon"] for c in conds]
    total = [RESULTS[c]["train_total"] for c in conds]

    ax.bar([i - bar_w for i in x], action, bar_w, label="L_action", color="#1f77b4")
    ax.bar(x, recon, bar_w, label="L_recon", color="#ff7f0e")
    ax.bar([i + bar_w for i in x], total, bar_w, label="L_total", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels([RESULTS[c]["label"] for c in conds], rotation=15)
    _setup(ax, "Loss (final logged step)", "Figure 3. Training losses at end of run (single seed = 42)")
    ax.legend(frameon=False, loc="upper left")

    out = OUT / "fig3_train_loss.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure4_per_dof_mae() -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    dofs = [r"$\Delta x$", r"$\Delta y$", r"$\Delta z$",
            r"$\Delta R$", r"$\Delta P$", r"$\Delta Y$", "gripper"]
    x = list(range(len(dofs)))
    n_conds = len(RESULTS)
    bar_w = 0.85 / n_conds
    for i, (cond, info) in enumerate(RESULTS.items()):
        offset = (i - (n_conds - 1) / 2) * bar_w
        ax.bar([xi + offset for xi in x], info["val_mae_dim"], bar_w,
               label=info["label"], color=COLOURS[cond])
    ax.set_xticks(x)
    ax.set_xticklabels(dofs)
    _setup(ax, "Validation MAE", "Figure 4. Per-DoF validation MAE by condition (lower is better)")
    ax.set_ylim(0.74, 0.90)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))

    out = OUT / "fig4_per_dof_mae.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure5_val_total() -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    conds = list(RESULTS.keys())
    vals = [RESULTS[c]["val_total"] for c in conds]
    bars = ax.bar(conds, vals, color=[COLOURS[c] for c in conds])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    _setup(ax, "Validation total loss", "Figure 5. Validation total loss by condition")
    ax.set_ylim(0, max(vals) * 1.15)
    out = OUT / "fig5_val_total.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    p3 = figure3_train_loss()
    p4 = figure4_per_dof_mae()
    p5 = figure5_val_total()
    for p in (p3, p4, p5):
        print("Wrote", p.relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
    main()
