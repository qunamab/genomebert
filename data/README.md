# Data Directory — GenomeBERT

This directory contains all data-related scripts and dataset files.

## Structure

```
data/
├── download_data.py    ← Download & assemble datasets
├── preprocess.py       ← QC filtering and feature extraction
├── raw/                ← Raw downloaded files (auto-created)
│   ├── epd_human_promoters.fa.gz
│   └── ucsc_cre.txt.gz
└── processed/          ← Cleaned, model-ready CSV (auto-created)
    └── sequences.csv
```

## Dataset Sources

| Class | Source | Motif Signal |
|---|---|---|
| **Promoter** | [EPD Human Promoters](https://epd.expasy.org/) | TATAAAA (TATA box) |
| **Enhancer** | Synthetic (SP1 motif) | GGGCGG |
| **Binding Site** | Synthetic (E-box) | CACGTG |
| **Non-functional** | Random GC-balanced DNA | — |

## Quick Start

```bash
# Download and prepare all data
python data/download_data.py --dataset all --output data/

# QC-filter the processed CSV
python data/preprocess.py \
    --input  data/processed/sequences.csv \
    --output data/processed/sequences_clean.csv \
    --min_len 50 --max_len 500
```

## `sequences.csv` Format

| Column | Type | Description |
|---|---|---|
| `sequence` | str | DNA sequence (A/T/G/C/N only) |
| `label` | int | 0=Promoter, 1=Enhancer, 2=Binding, 3=Non-functional |
| `label_name` | str | Human-readable class name |

## Notes

- Real EPD promoters require internet access on first run.
- Enhancer and binding-site sequences are synthetic with planted motifs —
  they are sufficient for demonstrating the model but should be replaced
  with ENCODE ChIP-seq data for publication-quality results.
- All datasets are open-access and free for research use.
