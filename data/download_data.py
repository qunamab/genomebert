"""
download_data.py
================
Downloads open-source genomic datasets for GenomeBERT training.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
import urllib.request
import urllib.error

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────
DATASETS = {
    # EPD — human non-redundant promoters (+/- 250 bp around TSS)
    "epd_human_promoters": {
        "url":  "https://epd.expasy.org/ftp/epdnew/H_sapiens/006/Hs_EPDnew.fa.gz",
        "label":    "promoter",
        "filename": "epd_human_promoters.fa.gz",
        "description": "EPD Human Promoters (Hs_EPDnew)",
    },
    # UCSC hg38 ENCODE cCRE (regulatory element) BED file
    "ucsc_cre": {
        "url":  "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/encodeCcreCombined.txt.gz",
        "label":    "enhancer",
        "filename": "ucsc_cre.txt.gz",
        "description": "UCSC hg38 ENCODE Candidate Cis-Regulatory Elements",
    },
}

# UCSC DAS / REST (for extracting FASTA from coordinates)
UCSC_DAS_URL = "https://genome.ucsc.edu/cgi-bin/das/hg38/dna?segment={chrom}:{start},{end}"


# ─────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────
def _download_file(url: str, dest: Path, retries: int = 3) -> None:
    """Download a file with progress indicator and retry logic."""
    if dest.exists():
        logger.info(f"  Already downloaded: {dest.name} — skipping.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading {dest.name} ...")
    logger.info(f"  Source: {url}")

    for attempt in range(1, retries + 1):
        try:
            def _reporthook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100, downloaded * 100 / total_size)
                    mb = downloaded / 1e6
                    print(f"\r  Progress: {pct:5.1f}%  ({mb:.1f} MB)", end="", flush=True)

            urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)
            print()   # newline after progress
            logger.info(f"  ✓ Saved to {dest}")
            return
        except urllib.error.URLError as e:
            logger.warning(f"  Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def _read_fasta_gz(path: Path) -> List[Tuple[str, str]]:
    """Read a gzipped FASTA file → list of (header, sequence)."""
    records = []
    with gzip.open(str(path), "rt") as f:
        header, seq_parts = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
        if header:
            records.append((header, "".join(seq_parts)))
    return records


# ─────────────────────────────────────────────
# Per-source parsers
# ─────────────────────────────────────────────
def prepare_promoters(raw_dir: Path, out_dir: Path) -> List[Tuple[str, str]]:
    """Parse EPD promoter FASTA → [(sequence, label)]."""
    dest = raw_dir / DATASETS["epd_human_promoters"]["filename"]
    _download_file(DATASETS["epd_human_promoters"]["url"], dest)

    records = _read_fasta_gz(dest)
    logger.info(f"  EPD promoters: {len(records)} sequences")
    return [(seq, "promoter") for _, seq in records if len(seq) >= 50]


def generate_negative_sequences(
    n: int,
    length_range: Tuple[int, int] = (100, 500),
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Generate random DNA sequences as 'non-functional' negatives.
    Uses realistic GC-content distribution (~45-55%).
    """
    rng = np.random.default_rng(seed)
    bases = np.array(["A", "T", "G", "C"])
    # Biased toward GC ~50%
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    sequences = []
    for _ in range(n):
        length = rng.integers(*length_range)
        seq = "".join(rng.choice(bases, size=length, p=probs))
        sequences.append((seq, "non_functional"))
    logger.info(f"  Generated {n} random non-functional sequences")
    return sequences


def generate_synthetic_enhancers(
    n: int,
    length: int = 200,
    seed: int = 99,
) -> List[Tuple[str, str]]:
    """
    Generate synthetic enhancer-like sequences with SP1 motif (GGGCGG) planted.
    For demonstration when real ENCODE data is unavailable.
    """
    rng = np.random.default_rng(seed)
    bases = list("ATGC")
    MOTIF = "GGGCGG"
    sequences = []
    for _ in range(n):
        seq = list(rng.choice(list("ATGC"), size=length))
        pos = rng.integers(10, length - len(MOTIF) - 10)
        for j, ch in enumerate(MOTIF):
            seq[pos + j] = ch
        sequences.append(("".join(seq), "enhancer"))
    logger.info(f"  Generated {n} synthetic enhancer sequences")
    return sequences


def generate_synthetic_binding_sites(
    n: int,
    length: int = 150,
    seed: int = 77,
) -> List[Tuple[str, str]]:
    """Synthetic binding site sequences with E-box (CACGTG) motif."""
    rng = np.random.default_rng(seed)
    MOTIF = "CACGTG"
    sequences = []
    for _ in range(n):
        seq = list(rng.choice(list("ATGC"), size=length))
        pos = rng.integers(10, length - len(MOTIF) - 10)
        for j, ch in enumerate(MOTIF):
            seq[pos + j] = ch
        sequences.append(("".join(seq), "binding_site"))
    logger.info(f"  Generated {n} synthetic binding site sequences")
    return sequences


# ─────────────────────────────────────────────
# Save to CSV
# ─────────────────────────────────────────────
def save_dataset(
    data: List[Tuple[str, str]],
    out_dir: Path,
    filename: str = "sequences.csv",
    min_len: int = 50,
    max_len: int = 1000,
) -> None:
    """Clean and save dataset to CSV."""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    label_map = {
        "promoter":       0,
        "enhancer":       1,
        "binding_site":   2,
        "non_functional": 3,
    }

    n_written = 0
    n_skipped = 0
    label_counts = {k: 0 for k in label_map}

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "label", "label_name"])
        for seq, label_name in data:
            seq = seq.upper().replace(" ", "").replace("\n", "")
            # Filter non-ATGCN characters
            seq_clean = "".join(c for c in seq if c in "ATGCN")
            if not (min_len <= len(seq_clean) <= max_len):
                n_skipped += 1
                continue
            label_id = label_map.get(label_name, -1)
            if label_id == -1:
                n_skipped += 1
                continue
            writer.writerow([seq_clean, label_id, label_name])
            n_written += 1
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

    logger.info(f"\n  Dataset saved → {out_path}")
    logger.info(f"  Total: {n_written} samples  ({n_skipped} filtered)")
    for lbl, cnt in label_counts.items():
        logger.info(f"    {lbl:<20} : {cnt}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    raw_dir  = Path(args.output) / "raw"
    proc_dir = Path(args.output) / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    all_data: List[Tuple[str, str]] = []

    logger.info("\n" + "═" * 56)
    logger.info("  GenomeBERT — Dataset Preparation")
    logger.info("═" * 56)

    # ── Promoters
    if args.dataset in ("all", "promoters", "ucsc_promoters"):
        logger.info("\n[1/4] EPD Human Promoters ...")
        try:
            promoters = prepare_promoters(raw_dir, proc_dir)
            # Use at most 2000 for balanced dataset
            all_data.extend(promoters[:2000])
        except Exception as e:
            logger.warning(f"  Could not download real promoters ({e}). Using synthetic.")
            # Synthetic TATA-box promoters
            rng = np.random.default_rng(0)
            for _ in range(2000):
                seq = "".join(rng.choice(list("ATGC"), size=200))
                pos = rng.integers(30, 60)
                seq = seq[:pos] + "TATAAAA" + seq[pos+7:]
                all_data.append((seq, "promoter"))

    # ── Enhancers (synthetic)
    if args.dataset in ("all", "enhancers"):
        logger.info("\n[2/4] Synthetic Enhancer Sequences ...")
        all_data.extend(generate_synthetic_enhancers(2000))

    # ── Binding Sites (synthetic)
    if args.dataset in ("all", "binding_sites"):
        logger.info("\n[3/4] Synthetic Protein-Binding Site Sequences ...")
        all_data.extend(generate_synthetic_binding_sites(2000))

    # ── Negatives
    if args.dataset in ("all", "negatives"):
        logger.info("\n[4/4] Random Non-functional Sequences ...")
        all_data.extend(generate_negative_sequences(2000))

    # ── Shuffle & save
    import random
    random.seed(42)
    random.shuffle(all_data)

    save_dataset(all_data, proc_dir, filename="sequences.csv")
    logger.info("\n✓ Dataset preparation complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and prepare genomic datasets for GenomeBERT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        choices=["all", "promoters", "ucsc_promoters", "enhancers",
                 "binding_sites", "negatives"],
        default="all",
    )
    p.add_argument("--output", type=str, default="data/",
                   help="Base output directory")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
