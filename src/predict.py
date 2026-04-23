"""
predict.py
==========
Inference utilities for GenomeBERT.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import KmerTokenizer
from model import load_checkpoint

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Promoter", "Enhancer", "Binding Site", "Non-functional"]
EMOJI      = ["🔬", "✨", "🔗", "➖"]

# ─────────────────────────────────────────────────────────────────────
# ASCII bar visualisation
# ─────────────────────────────────────────────────────────────────────
def _prob_bar(prob: float, width: int = 30) -> str:
    filled = int(round(prob * width))
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────────────────────────────
def predict_sequence(
    sequence: str,
    checkpoint_path: str,
    kmer: int = 6,
    max_length: int = 512,
    device: str = "cpu",
    topk: int = 4,
) -> Dict:
    """
    Run inference on a single DNA sequence.

    Returns
    -------
    dict with keys: predicted_class, probabilities, sequence_info
    """
    device = torch.device(device)

    # Load model
    model, saved_metrics = load_checkpoint(checkpoint_path, device=str(device))
    model.eval()

    # Tokenise
    tokenizer = KmerTokenizer(k=kmer, max_length=max_length)
    encoded = tokenizer.encode(sequence.upper())

    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long).to(device)
    attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long).to(device)

    # Forward pass
    with torch.no_grad():
        out = model(input_ids, attention_mask)
        probs = F.softmax(out["logits"], dim=-1).squeeze().cpu().numpy()

    pred_class = int(probs.argmax())
    results = {
        "sequence": sequence,
        "length": len(sequence),
        "num_kmers": len(tokenizer.sequence_to_kmers(sequence)),
        "predicted_class": CLASS_NAMES[pred_class],
        "predicted_id": pred_class,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "confidence": float(probs[pred_class]),
        "model_metrics": saved_metrics,
    }
    return results


# ─────────────────────────────────────────────────────────────────────
# Pretty console output
# ─────────────────────────────────────────────────────────────────────
def print_results(results: Dict) -> None:
    seq = results["sequence"]
    preview = seq[:40] + ("..." if len(seq) > 40 else "")

    print("\n" + "═" * 62)
    print("  🧬  GenomeBERT — Sequence Inference Result")
    print("═" * 62)
    print(f"  Sequence  : {preview}")
    print(f"  Length    : {results['length']} nt  ({results['num_kmers']} k-mers)")
    print("─" * 62)
    print(f"  Prediction: {EMOJI[results['predicted_id']]}  {results['predicted_class'].upper()}")
    print(f"  Confidence: {results['confidence']*100:.1f}%")
    print("─" * 62)
    print("  Class Probabilities:")
    for cls, prob in results["probabilities"].items():
        idx = CLASS_NAMES.index(cls)
        bar = _prob_bar(prob)
        print(f"  {EMOJI[idx]} {cls:<18} {bar}  {prob*100:5.1f}%")
    print("═" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────
# Batch prediction from FASTA / plain text file
# ─────────────────────────────────────────────────────────────────────
def predict_file(
    filepath: str,
    checkpoint_path: str,
    output_csv: str = "predictions.csv",
    kmer: int = 6,
    max_length: int = 512,
) -> None:
    """Predict all sequences in a FASTA or plain-text file."""
    import pandas as pd
    from Bio import SeqIO

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    sequences: List[str] = []
    seq_ids: List[str]   = []

    if filepath.suffix in (".fa", ".fasta"):
        for rec in SeqIO.parse(str(filepath), "fasta"):
            seq_ids.append(rec.id)
            sequences.append(str(rec.seq))
    else:  # plain text — one sequence per line
        with open(filepath) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    seq_ids.append(f"seq_{i}")
                    sequences.append(line)

    logger.info(f"Predicting {len(sequences)} sequences from {filepath}")

    rows = []
    for sid, seq in zip(seq_ids, sequences):
        try:
            res = predict_sequence(seq, checkpoint_path, kmer, max_length)
            rows.append({
                "id":               sid,
                "sequence_preview": seq[:30] + "...",
                "length":           res["length"],
                "predicted_class":  res["predicted_class"],
                "confidence":       res["confidence"],
                **{f"prob_{k}": v for k, v in res["probabilities"].items()},
            })
        except Exception as e:
            logger.warning(f"  Failed on {sid}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"Predictions saved → {output_csv}")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GenomeBERT — Single or batch DNA sequence inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sequence",   type=str, help="Single DNA sequence string")
    grp.add_argument("--file",       type=str, help="FASTA or plain-text file of sequences")

    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to genomebert_best.pt checkpoint")
    p.add_argument("--kmer",         type=int,   default=6)
    p.add_argument("--max_length",   type=int,   default=512)
    p.add_argument("--output_csv",   type=str,   default="predictions.csv",
                   help="(Batch mode) Output CSV path")
    p.add_argument("--device",       type=str,   default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    if args.sequence:
        results = predict_sequence(
            args.sequence, args.checkpoint,
            kmer=args.kmer, max_length=args.max_length, device=args.device,
        )
        print_results(results)
    else:
        predict_file(
            args.file, args.checkpoint,
            output_csv=args.output_csv,
            kmer=args.kmer, max_length=args.max_length,
        )
