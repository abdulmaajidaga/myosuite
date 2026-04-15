"""
plot_loss_curves.py — Plot training/validation loss curves for a phase.

Shows:
  1. Val loss curves for all experiments in the phase (one line per config)
  2. LR schedule overlay (right axis) to show when learning rate decayed
  3. Stability check: ranking correlation between epochs 150-200 vs 100-150
     (high correlation = ordering stabilised = 200 epochs is sufficient)

Usage:
  python test/scripts/plot_loss_curves.py --phase A
  python test/scripts/plot_loss_curves.py --phase A --output test/output/figures/loss_curves_A.png
"""

import os, sys, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import spearmanr

TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, TEST_DIR)
from experiments import EXPERIMENTS

# Colour groups by stage (same stage = same hue, different aug = different shade)
STAGE_COLOURS = {
    "A0": "#4878CF",   # blue
    "A1": "#6ACC65",   # green
    "A2": "#D65F5F",   # red
    "A3": "#B47CC7",   # purple
    "A4": "#C4AD66",   # gold
    "B0": "#4878CF",
    "B1": "#6ACC65",
    "B2": "#D65F5F",
    "B3": "#B47CC7",
    "B4": "#C4AD66",
}
AUG_ALPHA = {"smote": 1.0, "dtw": 0.65, "linear": 0.35}
AUG_DASH  = {"smote": "-", "dtw": "--", "linear": ":"}


def load_history(exp_name: str) -> pd.DataFrame | None:
    path = os.path.join(TEST_DIR, "output", exp_name, "history.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["experiment"] = exp_name
    return df


def stability_check(histories: dict[str, pd.DataFrame], window: int = 50) -> float:
    """
    Check ranking stability: Spearman ρ between mean val loss in
    epochs [max-2*window : max-window] vs [max-window : max].
    ρ close to 1.0 means ranking order didn't change in the last window.
    """
    early_means, late_means = [], []
    names = []
    for name, df in histories.items():
        n = len(df)
        if n < 2 * window:
            continue
        early_means.append(df["val"].iloc[n - 2*window : n - window].mean())
        late_means.append(df["val"].iloc[n - window :].mean())
        names.append(name)
    if len(names) < 3:
        return float("nan")
    rho, _ = spearmanr(early_means, late_means)
    return rho


def plot_phase(phase: str, output_path: str | None = None):
    exp_names = [k for k in EXPERIMENTS if k.startswith(phase.upper())]
    if not exp_names:
        print(f"No experiments found for phase '{phase}'")
        return

    histories = {}
    missing   = []
    for name in exp_names:
        df = load_history(name)
        if df is not None:
            histories[name] = df
        else:
            missing.append(name)

    if not histories:
        print(f"No history.csv files found yet — run Phase {phase} first.")
        return

    if missing:
        print(f"Missing ({len(missing)} not yet run): {missing}")

    # ── Figure layout ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss, ax_lr = axes

    # Panel 1: validation loss curves
    ax_loss.set_title(f"Phase {phase.upper()} — Validation Loss", fontsize=13)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Validation Loss")

    for name, df in sorted(histories.items()):
        # derive stage prefix and aug
        parts = name.split("_")
        stage_key = parts[0]          # e.g. "A3"
        aug_key   = parts[-1]         # e.g. "smote"
        colour = STAGE_COLOURS.get(stage_key, "#888888")
        alpha  = AUG_ALPHA.get(aug_key, 0.8)
        ls     = AUG_DASH.get(aug_key, "-")
        ax_loss.plot(df["epoch"], df["val"],
                     color=colour, alpha=alpha, linestyle=ls,
                     linewidth=1.5, label=name)

    ax_loss.legend(fontsize=7, ncol=2, loc="upper right")

    # Shade last 50 epochs (stability window)
    max_epoch = max(len(df) for df in histories.values())
    if max_epoch > 50:
        ax_loss.axvspan(max_epoch - 50, max_epoch, alpha=0.07,
                        color="black", label="stability window")

    # Panel 2: LR schedules
    ax_lr.set_title(f"Phase {phase.upper()} — Learning Rate Schedule", fontsize=13)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_yscale("log")

    for name, df in sorted(histories.items()):
        if "lr" not in df.columns:
            continue
        parts = name.split("_")
        stage_key = parts[0]
        aug_key   = parts[-1]
        colour = STAGE_COLOURS.get(stage_key, "#888888")
        alpha  = AUG_ALPHA.get(aug_key, 0.8)
        ls     = AUG_DASH.get(aug_key, "-")
        ax_lr.plot(df["epoch"], df["lr"],
                   color=colour, alpha=alpha, linestyle=ls,
                   linewidth=1.5, label=name)

    ax_lr.legend(fontsize=7, ncol=2, loc="lower left")

    # ── Stability annotation ────────────────────────────────────────────────
    rho = stability_check(histories)
    status = f"Rank stability (last-50 vs prev-50): ρ = {rho:.3f}"
    if not np.isnan(rho):
        colour = "green" if rho > 0.95 else ("orange" if rho > 0.85 else "red")
        fig.text(0.5, 0.01, status, ha="center", fontsize=10, color=colour,
                 style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        default = os.path.join(TEST_DIR, "output", "figures",
                               f"loss_curves_{phase.upper()}.png")
        os.makedirs(os.path.dirname(default), exist_ok=True)
        fig.savefig(default, dpi=150, bbox_inches="tight")
        print(f"Saved: {default}")

    plt.close(fig)

    # ── Console summary ─────────────────────────────────────────────────────
    print(f"\nPhase {phase.upper()} — final val loss summary ({len(histories)} completed):")
    rows = []
    for name, df in sorted(histories.items()):
        rows.append({
            "experiment": name,
            "final_val":  round(df["val"].iloc[-1], 4),
            "best_val":   round(df["val"].min(), 4),
            "best_epoch": int(df.loc[df["val"].idxmin(), "epoch"]),
            "final_lr":   f"{df['lr'].iloc[-1]:.2e}" if "lr" in df.columns else "—",
        })
    summary = pd.DataFrame(rows).sort_values("best_val")
    print(summary.to_string(index=False))
    print(f"\n{status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",  required=True, help="Phase letter: A or B")
    parser.add_argument("--output", default=None,  help="Output PNG path")
    args = parser.parse_args()
    plot_phase(args.phase, args.output)
