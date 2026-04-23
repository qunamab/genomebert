"""
preprocess.py
=============
Sequence cleaning and quality filtering utilities.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_BASES = set("ATGCN")


# ─────────────────────────────────────────────
# Sequence quality checks
# ─────────────────────────────────────────────
def normalize_sequence(seq: str) -> str:
    """Uppercase, replace U→T, strip whitespace."""
    return seq.upper().replace("U", "T").replace(" ", "").replace("\n", "")


def filter_ambiguous(seq: str, max_n_ratio: float = 0.1) -> bool:
    """Return True if sequence passes (has few N's)."""
    n_count = seq.count("N")
    return (n_count / max(len(seq), 1)) <= max_n_ratio


def gc_content(seq: str) -> float:
    """Compute GC fraction."""
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return gc / max(len(seq), 1)


def linguistic_complexity(seq: str, k: int = 4) -> float:
    """
    Measure sequence complexity as ratio of observed / possible k-mers.
    Low complexity sequences (repeats) score near 0.
    """
    if len(seq) < k:
        return 0.0
    kmers = {seq[i:i+k] for i in range(len(seq) - k + 1)}
    max_possible = min(4**k, len(seq) - k + 1)
    return len(kmers) / max(max_possible, 1)


def passes_quality(
    seq: str,
    min_len: int = 50,
    max_len: int = 1000,
    max_n_ratio: float = 0.10,
    min_gc: float = 0.20,
    max_gc: float = 0.80,
    min_complexity: float = 0.2,
) -> Tuple[bool, str]:
    """
    Run all QC checks on a sequence.

    Returns
    -------
    (passes: bool, reason: str)
    """
    if len(seq) < min_len:
        return False, f"too_short ({len(seq)} < {min_len})"
    if len(seq) > max_len:
        return False, f"too_long ({len(seq)} > {max_len})"
    if not filter_ambiguous(seq, max_n_ratio):
        return False, "too_many_N"
    gc = gc_content(seq)
    if gc < min_gc:
        return False, f"low_gc ({gc:.2f} < {min_gc})"
    if gc > max_gc:
        return False, f"high_gc ({gc:.2f} > {max_gc})"
    cpx = linguistic_complexity(seq)
    if cpx < min_complexity:
        return False, f"low_complexity ({cpx:.2f} < {min_complexity})"
    return True, "ok"


# ─────────────────────────────────────────────
# Sliding-window chunking
# ─────────────────────────────────────────────
def chunk_sequence(
    seq: str,
    window: int = 500,
    stride: int = 250,
) -> List[str]:
    """
    Split a long sequence into overlapping windows.
    Useful for chromosome-scale inputs.
    """
    chunks = []
    for i in range(0, max(1, len(seq) - window + 1), stride):
        chunk = seq[i : i + window]
        if len(chunk) >= window // 2:
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────
# Batch preprocessing
# ─────────────────────────────────────────────
def preprocess_dataframe(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    label_col: str = "label",
    **qc_kwargs,
) -> pd.DataFrame:
    """
    Apply QC pipeline to an entire DataFrame.

    Returns cleaned DataFrame with added feature columns:
      gc_content, complexity, length
    """
    logger.info(f"Input: {len(df)} sequences")

    df = df.copy()
    df[sequence_col] = df[sequence_col].apply(normalize_sequence)

    # QC filter
    qc_results = df[sequence_col].apply(
        lambda s: passes_quality(s, **qc_kwargs)
    )
    df["qc_pass"]   = qc_results.apply(lambda r: r[0])
    df["qc_reason"] = qc_results.apply(lambda r: r[1])

    n_fail = (~df["qc_pass"]).sum()
    logger.info(f"QC failed: {n_fail} sequences")
    logger.info(f"Failure breakdown:\n{df[~df['qc_pass']]['qc_reason'].value_counts().to_string()}")

    df = df[df["qc_pass"]].drop(columns=["qc_pass", "qc_reason"])

    # Add features
    df["gc_content"]  = df[sequence_col].apply(gc_content)
    df["complexity"]  = df[sequence_col].apply(linguistic_complexity)
    df["length"]      = df[sequence_col].apply(len)

    logger.info(f"Output: {len(df)} sequences after QC")
    return df.reset_index(drop=True)


def preprocess_csv(
    input_path: str,
    output_path: str,
    sequence_col: str = "sequence",
    label_col: str = "label",
    **qc_kwargs,
) -> None:
    """Load, clean, and re-save a sequences CSV."""
    df = pd.read_csv(input_path)
    df_clean = preprocess_dataframe(df, sequence_col, label_col, **qc_kwargs)
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data → {output_path}")


# ─────────────────────────────────────────────
# Feature statistics report
# ─────────────────────────────────────────────
def print_dataset_stats(df: pd.DataFrame, label_names: Optional[Dict[int, str]] = None) -> None:
    """Print a detailed summary of the processed dataset."""
    if label_names is None:
        label_names = {0: "Promoter", 1: "Enhancer", 2: "Binding Site", 3: "Non-functional"}

    print("\n" + "═" * 56)
    print("  DATASET STATISTICS")
    print("═" * 56)
    print(f"  Total sequences : {len(df):,}")
    print(f"  Median length   : {df['length'].median():.0f} nt")
    print(f"  Mean GC content : {df['gc_content'].mean():.3f}")
    print(f"  Mean complexity : {df['complexity'].mean():.3f}")
    print("─" * 56)
    print("  Class distribution:")
    for lid, lname in sorted(label_names.items()):
        count = (df["label"] == lid).sum()
        pct   = count / len(df) * 100
        bar   = "█" * int(pct / 2)
        print(f"  {lname:<20} {count:>5}  ({pct:5.1f}%)  {bar}")
    print("═" * 56 + "\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    p = argparse.ArgumentParser(description="Preprocess genomic sequence CSV")
    p.add_argument("--input",  required=True,  help="Input CSV path")
    p.add_argument("--output", required=True,  help="Output CSV path")
    p.add_argument("--min_len",     type=int,   default=50)
    p.add_argument("--max_len",     type=int,   default=1000)
    p.add_argument("--max_n_ratio", type=float, default=0.10)
    p.add_argument("--min_gc",      type=float, default=0.20)
    p.add_argument("--max_gc",      type=float, default=0.80)
    args = p.parse_args()

    preprocess_csv(
        args.input, args.output,
        min_len=args.min_len, max_len=args.max_len,
        max_n_ratio=args.max_n_ratio,
        min_gc=args.min_gc, max_gc=args.max_gc,
    )
