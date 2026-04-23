"""
evaluate.py
===========
Evaluation and visualization utilities for GenomeBERT.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server/notebook use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Aesthetics
# ─────────────────────────────────────────────
PALETTE = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "accent1":  "#58a6ff",   # blue
    "accent2":  "#3fb950",   # green
    "accent3":  "#f78166",   # red / promoter
    "accent4":  "#d2a8ff",   # purple / enhancer
    "accent5":  "#ffa657",   # orange / binding
    "accent6":  "#8b949e",   # grey / non-functional
    "text":     "#e6edf3",
    "grid":     "#21262d",
}
CLASS_COLORS = [
    PALETTE["accent3"],   # Promoter
    PALETTE["accent4"],   # Enhancer
    PALETTE["accent5"],   # Binding Site
    PALETTE["accent6"],   # Non-functional
]
CLASS_NAMES = ["Promoter", "Enhancer", "Binding Site", "Non-functional"]


def _apply_dark_theme() -> None:
    """Apply a dark GitHub-inspired Matplotlib theme."""
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "grid.color":        PALETTE["grid"],
        "text.color":        PALETTE["text"],
        "legend.facecolor":  PALETTE["surface"],
        "legend.edgecolor":  PALETTE["grid"],
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "figure.dpi":        150,
    })


# ─────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true   : ground-truth labels (N,)
    y_pred   : predicted labels    (N,)
    y_prob   : softmax probabilities (N, C) — needed for ROC-AUC
    """
    metrics: Dict[str, float] = {
        "accuracy":       accuracy_score(y_true, y_pred),
        "f1_macro":       f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":    f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro":precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall_macro":   recall_score(y_true, y_pred,    average="macro",    zero_division=0),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        safe_name = name.lower().replace(" ", "_")
        if i < len(per_class_f1):
            metrics[f"f1_{safe_name}"] = float(per_class_f1[i])

    # ROC-AUC (macro, OvR)
    if y_prob is not None and num_classes > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except ValueError:
            pass  # Can fail if some classes absent in batch

    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a rich per-class classification report."""
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print("\n" + "═" * 60)
    print("  CLASSIFICATION REPORT")
    print("═" * 60)
    print(report)


# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    _apply_dark_theme()
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        linewidths=0.5,
        linecolor=PALETTE["grid"],
        ax=ax,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 12, "weight": "bold", "color": PALETTE["text"]},
    )
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label", labelpad=10)
    ax.set_title(
        "GenomeBERT — Confusion Matrix"
        + (" (Normalized)" if normalize else ""),
        pad=15, weight="bold",
    )
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200,
                    facecolor=PALETTE["bg"])
        logger.info(f"Confusion matrix saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
# Training Curves
# ─────────────────────────────────────────────
def plot_training_curves(
    history: Dict[str, list],
    save_path: Optional[str] = None,
) -> plt.Figure:
    _apply_dark_theme()
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("GenomeBERT — Training Curves", fontsize=16, weight="bold", y=1.02)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=PALETTE["accent1"],
            linewidth=2, label="Train", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"], color=PALETTE["accent3"],
            linewidth=2, label="Val", marker="o", markersize=3)
    ax.set_title("Cross-Entropy Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], color=PALETTE["accent1"],
            linewidth=2, label="Train", marker="o", markersize=3)
    ax.plot(epochs, history["val_acc"], color=PALETTE["accent2"],
            linewidth=2, label="Val", marker="o", markersize=3)
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)

    # F1
    ax = axes[2]
    ax.plot(epochs, history["train_f1"], color=PALETTE["accent4"],
            linewidth=2, label="Train F1", marker="o", markersize=3)
    ax.plot(epochs, history["val_f1"],   color=PALETTE["accent5"],
            linewidth=2, label="Val F1",   marker="o", markersize=3)
    ax.set_title("Macro F1 Score")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_facecolor(PALETTE["surface"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200,
                    facecolor=PALETTE["bg"])
        logger.info(f"Training curves saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
# Attention Visualisation
# ─────────────────────────────────────────────
def plot_attention_map(
    attention_weights: List,   # list of (B, heads, L, L) tensors
    token_labels: Optional[List[str]] = None,
    layer_idx: int = 0,
    head_idx: int = 0,
    sample_idx: int = 0,
    save_path: Optional[str] = None,
    max_tokens: int = 40,
) -> plt.Figure:
    """Visualise a single attention head as a heatmap."""
    import torch
    _apply_dark_theme()

    attn = attention_weights[layer_idx]   # (B, heads, L, L) or (B, L, L)
    if attn.dim() == 4:
        attn = attn[sample_idx, head_idx].cpu().float().numpy()   # (L, L)
    else:
        attn = attn[sample_idx].cpu().float().numpy()

    # Truncate for visibility
    attn = attn[:max_tokens, :max_tokens]

    if token_labels is None:
        token_labels = [f"t{i}" for i in range(attn.shape[0])]
    token_labels = token_labels[:max_tokens]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn, cmap="viridis", aspect="auto",
                   vmin=0, vmax=attn.max())
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Attention Weight")

    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(token_labels)))
    ax.set_yticklabels(token_labels, fontsize=7)

    ax.set_title(
        f"Attention Map — Layer {layer_idx + 1}, Head {head_idx + 1}",
        weight="bold", pad=15,
    )
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200,
                    facecolor=PALETTE["bg"])
        logger.info(f"Attention map saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
# Per-class bar chart
# ─────────────────────────────────────────────
def plot_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    _apply_dark_theme()
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        CLASS_NAMES[:len(per_class_f1)],
        per_class_f1,
        color=CLASS_COLORS[:len(per_class_f1)],
        edgecolor=PALETTE["grid"],
        linewidth=0.8,
        width=0.5,
    )
    for bar, val in zip(bars, per_class_f1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom",
            color=PALETTE["text"], fontsize=11, weight="bold",
        )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 Score")
    ax.set_title("GenomeBERT — Per-Class F1 Score", weight="bold", pad=15)
    ax.set_facecolor(PALETTE["surface"])
    ax.grid(True, axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200,
                    facecolor=PALETTE["bg"])
        logger.info(f"Per-class F1 plot saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
# Load history and plot from saved JSON
# ─────────────────────────────────────────────
def plot_from_history_file(
    history_path: str,
    results_dir: str = "results/",
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(history_path) as f:
        history = json.load(f)
    plot_training_curves(history, save_path=str(results_dir / "training_curves.png"))
    logger.info("Plots saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate fake history and plot it
    n = 25
    fake_hist = {
        "train_loss": (np.linspace(1.4, 0.25, n) + np.random.randn(n) * 0.02).tolist(),
        "val_loss":   (np.linspace(1.5, 0.35, n) + np.random.randn(n) * 0.04).tolist(),
        "train_acc":  (np.linspace(0.35, 0.91, n) + np.random.randn(n) * 0.01).tolist(),
        "val_acc":    (np.linspace(0.30, 0.88, n) + np.random.randn(n) * 0.02).tolist(),
        "train_f1":   (np.linspace(0.30, 0.90, n) + np.random.randn(n) * 0.01).tolist(),
        "val_f1":     (np.linspace(0.25, 0.87, n) + np.random.randn(n) * 0.02).tolist(),
    }
    Path("results").mkdir(exist_ok=True)
    plot_training_curves(fake_hist, save_path="results/training_curves.png")
    print("Training curves saved to results/training_curves.png")
