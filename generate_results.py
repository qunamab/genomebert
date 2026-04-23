"""
generate_results.py
===================
Visualizes GenomeBERT result figures from a saved checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "data"))

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    from tokenizer import KmerTokenizer
    from model import load_checkpoint
    from dataset import generate_synthetic_data, build_dataloaders, split_dataset, GenomicDataset
    from evaluate import (
        compute_metrics,
        plot_confusion_matrix,
        plot_training_curves,
        plot_per_class_f1,
        plot_attention_map,
        print_classification_report,
        CLASS_NAMES,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    # ── Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        logger.info("Run training first: python src/train.py")
        return

    model, _ = load_checkpoint(str(ckpt_path), device=str(device))
    model.eval()

    # ── Load history from checkpoint
    ckpt_raw = torch.load(ckpt_path, map_location=device)
    history  = ckpt_raw.get("history")

    if history and len(history.get("train_loss", [])) > 0:
        logger.info("Plotting training curves from checkpoint history...")
        plot_training_curves(history, save_path=str(results_dir / "training_curves.png"))
    else:
        logger.warning("No training history in checkpoint — skipping curves.")

    # ── Generate test data
    max_len = ckpt_raw["config"]["max_position_embeddings"]
    tokenizer = KmerTokenizer(k=6, max_length=max_len)
    seqs, labels = generate_synthetic_data(n_samples=500, seq_length=max_len//2, seed=7)
    _, _, test_d = split_dataset(seqs, labels)
    test_ds = GenomicDataset(*test_d, tokenizer, max_length=max_len, augment=False)

    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(ids, mask)
            probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(probs.argmax(-1))
            all_true.extend(batch["labels"].numpy())

    y_true  = np.array(all_true)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    # ── Metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)
    print_classification_report(y_true, y_pred)
    logger.info("Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k:<25} {v:.4f}")

    # ── Confusion matrix
    plot_confusion_matrix(y_true, y_pred,
                          save_path=str(results_dir / "confusion_matrix.png"))

    # ── Per-class F1
    plot_per_class_f1(y_true, y_pred,
                      save_path=str(results_dir / "per_class_f1.png"))

    # ── Attention map (first promoter in test set)
    promo_idx = next((i for i, l in enumerate(test_d[1]) if l == 0), 0)
    sample = test_ds[promo_idx]
    sample_ids  = sample["input_ids"].unsqueeze(0).to(device)
    sample_mask = sample["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(sample_ids, sample_mask)

    token_labels = [tokenizer.id2token.get(tid, "?") for tid in sample["input_ids"].tolist()]
    plot_attention_map(
        out["attention_weights"],
        token_labels=token_labels,
        layer_idx=0, head_idx=0, sample_idx=0,
        max_tokens=40,
        save_path=str(results_dir / "attention_visualization.png"),
    )

    logger.info(f"\n✓ All results saved to {results_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate GenomeBERT result figures")
    p.add_argument("--checkpoint",   default="checkpoints/genomebert_best.pt")
    p.add_argument("--results_dir",  default="results/")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
