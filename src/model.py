"""
model.py
========
GenomeBERT — Transformer Encoder for Genomic Sequence Classification.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────
@dataclass
class GenomeBERTConfig:
    """All hyper-parameters that define a GenomeBERT model."""

    vocab_size: int = 5**6 + 5          # 4^6 + special tokens (approx)
    hidden_size: int = 128              # Token embedding dimension
    num_hidden_layers: int = 4          # Number of Transformer encoder layers
    num_attention_heads: int = 8        # Multi-head attention heads
    intermediate_size: int = 512        # FFN inner dimension
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512  # Max sequence length
    num_labels: int = 4                 # Classes: Promoter/Enhancer/Binding/None
    classifier_dropout: float = 0.2
    label_names: list = field(
        default_factory=lambda: ["Promoter", "Enhancer", "Binding Site", "Non-functional"]
    )

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )


# ─────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────
class GenomeBERTEmbeddings(nn.Module):
    """Token embedding + sinusoidal positional encoding."""

    def __init__(self, config: GenomeBERTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=config.hidden_size,
            max_len=config.max_position_embeddings,
            dropout=config.hidden_dropout_prob,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)          # (B, L, D)
        x = self.positional_encoding(x)              # add pos encoding + dropout
        x = self.layer_norm(x)
        return x


# ─────────────────────────────────────────────────────────────────────
# Transformer Encoder Layer (manual, for transparency)
# ─────────────────────────────────────────────────────────────────────
class GenomeBERTLayer(nn.Module):
    """
    Single Transformer encoder layer:
      MultiHeadSelfAttention → Add & Norm → FFN → Add & Norm
    """

    def __init__(self, config: GenomeBERTConfig):
        super().__init__()
        d = config.hidden_size

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d, eps=1e-12)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, d),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffn_norm = nn.LayerNorm(d, eps=1e-12)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x                : (B, L, D)
        key_padding_mask : (B, L) — True positions are IGNORED by attention

        Returns
        -------
        (output, attention_weights)
        """
        attn_out, attn_weights = self.attention(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,   # keep per-head weights
        )
        x = self.attn_norm(x + attn_out)          # residual + norm
        x = self.ffn_norm(x + self.ffn(x))        # residual + norm
        return x, attn_weights


# ─────────────────────────────────────────────────────────────────────
# Full GenomeBERT Model
# ─────────────────────────────────────────────────────────────────────
class GenomeBERT(nn.Module):
    """
    Full GenomeBERT encoder + classification head.

    Forward pass returns a ModelOutput-like dict:
        logits              : (B, num_labels)
        loss                : scalar (if labels provided)
        hidden_states       : (B, L, D) — final encoder output
        attention_weights   : list of per-layer attention tensors
    """

    def __init__(self, config: GenomeBERTConfig):
        super().__init__()
        self.config = config

        self.embeddings = GenomeBERTEmbeddings(config)

        self.encoder = nn.ModuleList(
            [GenomeBERTLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size // 2, config.num_labels),
        )

        self._init_weights()
        logger.info(
            f"GenomeBERT initialised | "
            f"vocab={config.vocab_size}, d={config.hidden_size}, "
            f"layers={config.num_hidden_layers}, heads={config.num_attention_heads} | "
            f"params={self.num_parameters():,}"
        )

    # ------------------------------------------------------------------
    # Weight initialisation (BERT-style)
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Build key_padding_mask: True = pad (ignored by attention)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # (B, L)
        else:
            key_padding_mask = None

        # Embeddings
        hidden = self.embeddings(input_ids)          # (B, L, D)

        # Transformer encoder layers
        all_attn_weights = []
        for layer in self.encoder:
            hidden, attn_w = layer(hidden, key_padding_mask=key_padding_mask)
            all_attn_weights.append(attn_w)

        # [CLS] token → classifier
        cls_repr = hidden[:, 0, :]                   # (B, D)
        logits = self.classifier(cls_repr)           # (B, num_labels)

        # Loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden,
            "attention_weights": all_attn_weights,
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return softmax probabilities."""
        out = self.forward(input_ids, attention_mask)
        return F.softmax(out["logits"], dim=-1)

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return class indices."""
        return self.predict_proba(input_ids, attention_mask).argmax(dim=-1)


# ─────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────
def build_model(vocab_size: int, config_overrides: Optional[dict] = None) -> GenomeBERT:
    """Create and return a GenomeBERT model ready for training."""
    cfg = GenomeBERTConfig(vocab_size=vocab_size)
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)
    return GenomeBERT(cfg)


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[GenomeBERT, dict]:
    """Load a saved GenomeBERT checkpoint."""
    ckpt = torch.load(path, map_location=device)
    cfg = GenomeBERTConfig(**ckpt["config"])
    model = GenomeBERT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint from {path}")
    return model, ckpt.get("metrics", {})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = GenomeBERTConfig(vocab_size=5000)
    model = GenomeBERT(cfg)
    print(f"Model initialized with {model.num_parameters():,} parameters.")
