# Temporal-HL

Temporal-HL is a motion-aware symbolic extension of framewise hand notation. This repository contains:

- source code for Temporal-HL label generation
- training and evaluation scripts
- paper draft and experiment summaries
- lightweight paper tables and text assets

Heavy reproducibility assets are intentionally **not** tracked in Git. They must live under:

`temporal_hl_cache/`

## Reproducibility Layout

After cloning this repository, place the downloaded cache directory at:

```text
Temporal-HL/
├── README.md
├── temporal_hl/
├── train_temporal_hl.py
├── train_token_reconstruction.py
├── train_token_reconstruction_v2.py
├── ...
└── temporal_hl_cache/
    ├── data/
    ├── artifacts/
    ├── runs/
    └── paper_assets/
```

The current codebase assumes this layout by default.

## What Belongs In Git vs Cache

Tracked in Git:

- code
- scripts
- paper draft and summaries
- lightweight markdown tables

Stored only in `temporal_hl_cache/`:

- source data
- generated manifests and `.npz` files
- training checkpoints
- run logs
- generated bitmap figures

## Current Cached Assets Expected

- `temporal_hl_cache/data/`
- `temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json`
- `temporal_hl_cache/runs/`
- `temporal_hl_cache/paper_assets/`

## Quick Start

### 1. Preprocess

```bash
python -m temporal_hl.preprocess \
  --annotations temporal_hl_cache/data/train/annotations.jsonl \
  --output-root temporal_hl_cache/artifacts/temporal_hl
```

### 2. Train notation translation

```bash
python train_temporal_hl.py \
  --manifest temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json \
  --mode temporal \
  --save-dir temporal_hl_cache/runs/temporal_hl
```

### 3. Train token-to-motion reconstruction

```bash
python train_token_reconstruction_v2.py \
  --manifest temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json \
  --arch transformer \
  --mode temporal \
  --save-dir temporal_hl_cache/runs/token_recon_v2
```

## Important Notes

- The benchmark used in the paper is the reorganized clip-level split stored in the cache assets.
- Source metadata fields such as `source_split` are provenance only.
- The repository is organized so that an external machine can reproduce the current state by:
  1. cloning this repository
  2. downloading `temporal_hl_cache/`
  3. placing `temporal_hl_cache/` at the repo root

## Main Paper Files

- `paper_draft.md`
- `results_summary.md`
- `reviewer_risk.md`
- `mock_review.md`
- `paper_assets/main_tables.md`
- `paper_assets/appendix_tables.md`
