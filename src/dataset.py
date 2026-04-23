"""
dataset.py
==========
PyTorch Dataset & DataLoader utilities for GenomeBERT.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from tokenizer import KmerTokenizer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Label maps
# ─────────────────────────────────────────────
LABEL2ID: Dict[str, int] = {
    "promoter": 0,
    "enhancer": 1,
    "binding_site": 2,
    "non_functional": 3,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


# ─────────────────────────────────────────────
# Core Dataset
# ─────────────────────────────────────────────
class GenomicDataset(Dataset):
    """
    PyTorch Dataset for genomic sequence classification.

    Parameters
    ----------
    sequences : list of DNA strings
    labels    : list of integer class labels
    tokenizer : KmerTokenizer instance
    max_length: int — passed to tokenizer
    augment   : bool — random reverse complement augmentation
    """

    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        tokenizer: KmerTokenizer,
        max_length: int = 512,
        augment: bool = False,
    ):
        assert len(sequences) == len(labels), "sequences and labels must be same length"
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

        logger.info(
            f"GenomicDataset | samples={len(sequences)}, "
            f"augment={augment}, max_length={max_length}"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Optional augmentation: random reverse complement
        if self.augment and np.random.random() < 0.5:
            seq = self._reverse_complement(seq)

        encoded = self.tokenizer.encode(
            seq,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
        )

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Compute the reverse complement of a DNA sequence."""
        complement = str.maketrans("ATGCN", "TACGN")
        return seq.upper().translate(complement)[::-1]

    # ------------------------------------------------------------------
    # Class balance info
    # ------------------------------------------------------------------
    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted sampling."""
        counts = np.bincount(self.labels, minlength=len(LABEL2ID))
        counts = np.where(counts == 0, 1, counts)           # avoid div-by-zero
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float)

    def sample_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler."""
        cw = self.class_weights().numpy()
        return [cw[lbl] for lbl in self.labels]


# ─────────────────────────────────────────────
# CSV loader
# ─────────────────────────────────────────────
def load_csv(
    path: Union[str, Path],
    sequence_col: str = "sequence",
    label_col: str = "label",
    label_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[int]]:
    """
    Load a CSV file and return (sequences, labels).

    The label column may contain string names (mapped via label_map)
    or integer IDs directly.
    """
    df = pd.read_csv(path)
    required_cols = {sequence_col, label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    sequences = df[sequence_col].astype(str).tolist()

    if label_map is None:
        label_map = LABEL2ID

    if df[label_col].dtype == object:
        labels = [label_map[lbl.strip().lower()] for lbl in df[label_col]]
    else:
        labels = df[label_col].astype(int).tolist()

    logger.info(f"Loaded {len(sequences)} samples from {path}")
    return sequences, labels


# ─────────────────────────────────────────────
# Train/Val/Test split helper
# ─────────────────────────────────────────────
def split_dataset(
    sequences: List[str],
    labels: List[int],
    val_size: float = 0.10,
    test_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """
    Stratified train / val / test split.

    Returns
    -------
    (train_seqs, train_labels), (val_seqs, val_labels), (test_seqs, test_labels)
    """
    train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(
        sequences, labels,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=random_state,
    )
    relative_test = test_size / (val_size + test_size)
    val_seqs, test_seqs, val_labels, test_labels = train_test_split(
        temp_seqs, temp_labels,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=random_state,
    )

    for split, s, l in [
        ("Train", train_seqs, train_labels),
        ("Val  ", val_seqs, val_labels),
        ("Test ", test_seqs, test_labels),
    ]:
        logger.info(f"  {split}: {len(s)} samples | "
                    f"class dist: {np.bincount(l).tolist()}")

    return (
        (train_seqs, train_labels),
        (val_seqs, val_labels),
        (test_seqs, test_labels),
    )


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────
def build_dataloaders(
    train_data: Tuple[List[str], List[int]],
    val_data: Tuple[List[str], List[int]],
    test_data: Tuple[List[str], List[int]],
    tokenizer: KmerTokenizer,
    batch_size: int = 32,
    num_workers: int = 0,
    max_length: int = 512,
    use_weighted_sampling: bool = True,
) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders."""

    train_ds = GenomicDataset(*train_data, tokenizer, max_length, augment=True)
    val_ds   = GenomicDataset(*val_data,   tokenizer, max_length, augment=False)
    test_ds  = GenomicDataset(*test_data,  tokenizer, max_length, augment=False)

    # Weighted sampler for class imbalance
    train_sampler = None
    shuffle_train = True
    if use_weighted_sampling:
        sample_w = train_ds.sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_w,
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle_train = False   # sampler is mutually exclusive with shuffle

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }


# ─────────────────────────────────────────────
# Synthetic synthetic data generator
# ─────────────────────────────────────────────
def generate_synthetic_data(
    n_samples: int = 2000,
    seq_length: int = 200,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Generate synthetic DNA sequences with planted motifs for testing.

    Motifs:
      Promoter       → TATAAAA (TATA box)
      Enhancer       → GGGCGG  (SP1 site)
      Binding Site   → CACGTG  (E-box)
      Non-functional → random
    """
    rng = np.random.default_rng(seed)
    bases = list("ATGC")
    motifs = {
        0: "TATAAAA",   # Promoter
        1: "GGGCGG",    # Enhancer
        2: "CACGTG",    # Binding Site
        3: None,        # Non-functional
    }

    sequences, labels = [], []
    per_class = n_samples // 4

    for label, motif in motifs.items():
        for _ in range(per_class):
            seq = "".join(rng.choice(bases, size=seq_length))
            if motif:
                # Plant motif at a random position
                pos = rng.integers(0, seq_length - len(motif))
                seq = seq[:pos] + motif + seq[pos + len(motif):]
            sequences.append(seq.upper())
            labels.append(label)

    # Shuffle
    idx = np.random.permutation(len(sequences))
    return [sequences[i] for i in idx], [labels[i] for i in idx]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tokenizer = KmerTokenizer(k=6, max_length=128)
    seqs, lbls = generate_synthetic_data(200, seq_length=100)
    ds = GenomicDataset(seqs[:100], lbls[:100], tokenizer, max_length=128)
    batch = ds[0]
    print({k: v.shape for k, v in batch.items()})
