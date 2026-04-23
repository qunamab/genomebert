"""
train.py
========
Training loop for GenomeBERT model.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    TENSORBOARD_AVAILABLE = False
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import KmerTokenizer
from model import GenomeBERT, GenomeBERTConfig
from dataset import load_csv, split_dataset, build_dataloaders, generate_synthetic_data
from evaluate import compute_metrics

# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Learning-rate schedule: linear warmup + cosine decay
# ─────────────────────────────────────────────
def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────
def train_one_epoch(
    model: GenomeBERT,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Train", leave=False, ncols=90):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask, labels)
        loss = out["loss"]

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["loss"] = total_loss / len(loader)
    return metrics


@torch.no_grad()
def evaluate_epoch(
    model: GenomeBERT,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Eval ", leave=False, ncols=90):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids, attention_mask, labels)
        total_loss += out["loss"].item()

        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu
        else "mps" if torch.backends.mps.is_available() and not args.cpu
        else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Tokenizer
    tokenizer = KmerTokenizer(k=args.kmer, stride=1, max_length=args.max_length)

    # ── Data
    if args.data_dir and Path(args.data_dir).exists():
        seqs, labels = load_csv(
            Path(args.data_dir) / "sequences.csv",
            sequence_col="sequence",
            label_col="label",
        )
        logger.info(f"Loaded {len(seqs)} real samples from {args.data_dir}")
    else:
        logger.warning("No data_dir found — using SYNTHETIC synthetic data (2 000 samples).")
        seqs, labels = generate_synthetic_data(n_samples=2000, seq_length=200)

    (train_d, val_d, test_d) = split_dataset(seqs, labels)

    loaders = build_dataloaders(
        train_d, val_d, test_d,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_weighted_sampling=True,
    )

    # ── Model
    config = GenomeBERTConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.max_length,
        num_labels=4,
    )
    model = GenomeBERT(config).to(device)
    logger.info(f"Parameters: {model.num_parameters():,}")

    # ── Optimizer & Scheduler
    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ── Output Directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        tb_dir = output_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"TensorBoard logs → {tb_dir}")
    else:
        logger.warning("TensorBoard unavailable (protobuf/compatibility issue). Training will proceed without event logging.")

    # ── Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
        "train_f1": [],   "val_f1": [],
    }

    logger.info(f"\n{'═'*60}")
    logger.info(f"  GenomeBERT Training — {args.epochs} epochs")
    logger.info(f"{'═'*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, loaders["train"], optimizer, scheduler, device, args.grad_clip
        )
        val_m = evaluate_epoch(model, loaders["val"], device)

        elapsed = time.time() - t0

        # Log
        if writer:
            for k, v in train_m.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_m.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(train_m.get("accuracy", 0))
        history["val_acc"].append(val_m.get("accuracy", 0))
        history["train_f1"].append(train_m.get("f1_macro", 0))
        history["val_f1"].append(val_m.get("f1_macro", 0))

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss={train_m['loss']:.4f}  acc={train_m.get('accuracy',0):.3f}  "
            f"f1={train_m.get('f1_macro',0):.3f} | "
            f"Val   loss={val_m['loss']:.4f}  acc={val_m.get('accuracy',0):.3f}  "
            f"f1={val_m.get('f1_macro',0):.3f} | "
            f"{elapsed:.1f}s"
        )

        # ── Checkpoint (best val loss)
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            patience_counter = 0
            ckpt_path = output_dir / "genomebert_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "config": {
                        "vocab_size": config.vocab_size,
                        "hidden_size": config.hidden_size,
                        "num_hidden_layers": config.num_hidden_layers,
                        "num_attention_heads": config.num_attention_heads,
                        "intermediate_size": config.intermediate_size,
                        "max_position_embeddings": config.max_position_embeddings,
                        "num_labels": config.num_labels,
                    },
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": {**train_m, **{f"val_{k}": v for k, v in val_m.items()}},
                    "history": history,
                },
                ckpt_path,
            )
            logger.info(f"  ✓ Checkpoint saved (val_loss={best_val_loss:.4f}) → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"  Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

    if writer:
        writer.close()

    # ── Final test evaluation
    logger.info("\nRunning final test-set evaluation...")
    best_model_path = output_dir / "genomebert_best.pt"
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate_epoch(model, loaders["test"], device)

    logger.info(
        f"\n{'═'*60}\n"
        f"  TEST RESULTS\n"
        f"  Loss     : {test_m['loss']:.4f}\n"
        f"  Accuracy : {test_m.get('accuracy', 0):.4f}\n"
        f"  F1 Macro : {test_m.get('f1_macro', 0):.4f}\n"
        f"{'═'*60}"
    )

    # Save history for plotting
    import json
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Artifacts saved to {output_dir}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GenomeBERT on genomic sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",    type=str,   default=None,  help="Directory with sequences.csv")
    p.add_argument("--output_dir",  type=str,   default="checkpoints/", help="Where to save checkpoints")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--kmer",        type=int,   default=6)
    p.add_argument("--max_length",  type=int,   default=512)
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--num_layers",  type=int,   default=4)
    p.add_argument("--num_heads",   type=int,   default=8)
    p.add_argument("--weight_decay",type=float, default=1e-2)
    p.add_argument("--warmup_ratio",type=float, default=0.06)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--patience",    type=int,   default=7,     help="Early-stopping patience")
    p.add_argument("--cpu",         action="store_true",       help="Force CPU training")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
