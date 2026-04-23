"""
tokenizer.py
============
k-mer DNA Tokenizer for GenomeBERT.
"""

from __future__ import annotations

import itertools
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
NUCLEOTIDES = ["A", "T", "G", "C", "N"]
SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[UNK]": 3,
    "[MASK]": 4,
}


# ─────────────────────────────────────────────
# KmerTokenizer
# ─────────────────────────────────────────────
class KmerTokenizer:
    """
    Tokenizes DNA/RNA sequences using a sliding k-mer window.

    Parameters
    ----------
    k : int
        Length of each k-mer (default: 6).
    stride : int
        Step size of the sliding window (default: 1 for full overlap).
    max_length : int
        Maximum total token sequence length (including [CLS] and [SEP]).
    """

    def __init__(self, k: int = 6, stride: int = 1, max_length: int = 512):
        self.k = k
        self.stride = stride
        self.max_length = max_length

        # Build vocabulary
        self.vocab: Dict[str, int] = dict(SPECIAL_TOKENS)
        self._build_vocab()

        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.unk_id = self.vocab["[UNK]"]
        self.mask_id = self.vocab["[MASK]"]

        logger.info(
            f"KmerTokenizer initialised | k={k}, stride={stride}, "
            f"vocab_size={len(self.vocab)}, max_length={max_length}"
        )

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------
    def _build_vocab(self) -> None:
        """Generate all possible k-mer combinations from {A, T, G, C, N}."""
        next_id = len(self.vocab)
        for kmer in itertools.product(NUCLEOTIDES, repeat=self.k):
            token = "".join(kmer)
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1

        # Reverse mapping
        self.id2token: Dict[int, str] = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def sequence_to_kmers(self, sequence: str) -> List[str]:
        """Convert a nucleotide string into a list of k-mer strings."""
        sequence = sequence.upper().replace("U", "T")  # RNA → DNA
        kmers: List[str] = []
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmer = sequence[i : i + self.k]
            # Replace invalid chars with N
            kmer = "".join(c if c in NUCLEOTIDES else "N" for c in kmer)
            kmers.append(kmer)
        return kmers

    def encode(
        self,
        sequence: str,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True,
    ) -> Dict[str, List[int]]:
        """
        Encode a single DNA sequence.

        Returns
        -------
        dict with keys:
            - input_ids       : token ID list
            - attention_mask  : 1 for real tokens, 0 for padding
        """
        kmers = self.sequence_to_kmers(sequence)

        # Truncate to fit within max_length - 2 (for [CLS] + [SEP])
        max_kmers = self.max_length - 2 if add_special_tokens else self.max_length
        if truncation and len(kmers) > max_kmers:
            kmers = kmers[:max_kmers]

        token_ids = [self.vocab.get(km, self.unk_id) for km in kmers]

        if add_special_tokens:
            token_ids = [self.cls_id] + token_ids + [self.sep_id]

        attention_mask = [1] * len(token_ids)

        # Pad
        if padding:
            pad_len = self.max_length - len(token_ids)
            token_ids = token_ids + [self.pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        result = {"input_ids": token_ids}
        if return_attention_mask:
            result["attention_mask"] = attention_mask

        return result

    def encode_batch(
        self,
        sequences: List[str],
        **kwargs,
    ) -> Dict[str, List[List[int]]]:
        """Encode a list of sequences as a batch."""
        encoded = [self.encode(seq, **kwargs) for seq in sequences]
        return {
            key: [item[key] for item in encoded]
            for key in encoded[0]
        }

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert a list of token IDs back to a (reconstructed) sequence string."""
        tokens = [self.id2token.get(tid, "[UNK]") for tid in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        # Re-stitch using first character of each k-mer
        if not tokens:
            return ""
        sequence = tokens[0]
        for t in tokens[1:]:
            sequence += t[-1]  # append last char (stride=1 assumption)
        return sequence

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save vocab to a JSON file."""
        import json
        with open(path, "w") as f:
            json.dump({"k": self.k, "stride": self.stride,
                       "max_length": self.max_length,
                       "vocab": self.vocab}, f, indent=2)
        logger.info(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "KmerTokenizer":
        """Load tokenizer from a saved JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        tok = cls(k=data["k"], stride=data["stride"],
                  max_length=data["max_length"])
        tok.vocab = {k: int(v) for k, v in data["vocab"].items()}
        tok.id2token = {int(v): k for k, v in tok.vocab.items()}
        return tok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tokenizer = KmerTokenizer(k=6, stride=1, max_length=128)
    seq = "ATGCTAGCTAGCATCGATCGATCGATCG"
    encoded = tokenizer.encode(seq)
    print(f"Vocab size  : {tokenizer.vocab_size}")
