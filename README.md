# Curvature Subspace Compression

<p align="center">
  <img src="https://github.com/user-attachments/assets/2b893f84-c708-4fb2-b6d8-329b34622473" width="416" height="207" />
</p>

A method for compressing transformer MLP blocks using curvature-aligned subspace decomposition, reducing memorization while preserving (and sometimes improving) generalization.

## Overview

This repository implements **curvature subspace compression** for transformer models, replacing MLP weight matrices with a three-factor approximation (`W ≈ L @ C @ R.T`) aligned to low-curvature directions of the loss landscape. The method:

- Reduces memorization (as measured by synthetic canary perplexity) with minimal impact on validation perplexity
- Can achieve better validation perplexity at a fixed parameter budget compared to uncompressed baselines
- Uses Kronecker-factored approximate curvature (K-FAC) to identify low-curvature subspaces


<p align="center">
  <img src="https://github.com/user-attachments/assets/8442615c-d133-421e-8d87-675d80bbf297"  width="416" height="207" />
</p>

## Installation

```bash
pip install torch transformers datasets fastapi uvicorn matplotlib numpy
```

## Quick Start

### Training a Baseline Model

```bash
python -m curvature_subspace_compression.memorization_kfac.experiments.run_crt_tiny_transformer \
  --model-dim 320 \
  --n-layers 5 \
  --n-heads 5 \
  --epochs 1 \
  --save-model-path artifacts/baseline.pt
```

### Running Compression Experiments

```bash
python -m curvature_subspace_compression.memorization_kfac.experiments.run_curvature_subspace_tiny_transformer \
  --checkpoint artifacts/baseline.pt \
  --keep-fractions 1.0 0.5 0.2 0.1 0.05 0.02 \
  --output-dir artifacts/compression_sweep
```

### Plotting Results

```bash
# Validation vs canary perplexity across compression levels
python -m curvature_subspace_compression.memorization_kfac.experiments.plot_additional_metrics \
  --base-dir artifacts/compression_sweep \
  --output artifacts/plots/additional_metrics.png

# Validation perplexity vs parameter count
python -m curvature_subspace_compression.memorization_kfac.experiments.plot_val_perplexity_vs_params \
  --base-dir artifacts/compression_sweep \
  --output artifacts/plots/val_perplexity_vs_params.png
```

### Deploying Models as API

```bash
# Export compressed checkpoint
python -m curvature_subspace_compression.api.export_compressed_checkpoint \
  --comp-json artifacts/tiny_dim320_comp020.json \
  --out artifacts/tiny_dim320_comp020.pt

# Start API server
python -m uvicorn curvature_subspace_compression.api.server:app --host 0.0.0.0 --port 8000
```

See `api/README.md` for API usage details.

## Key Components

- **`memorization_kfac/subspace_linear.py`**: `CurvatureSubspaceLinear` module implementing the three-factor decomposition
- **`memorization_kfac/experiments/run_curvature_subspace_tiny_transformer.py`**: Main compression experiment script
- **`memorization_kfac/experiments/run_crt_tiny_transformer.py`**: Baseline training script
- **`api/`**: FastAPI service for deploying baseline and compressed models side-by-side

## Citation

If you use this code, please cite:

```bibtex
@misc{curvature_subspace_compression,
  title={Curvature Subspace Compression for Transformer MLPs},
  author={Your Name},
  year={2025},
  url={https://github.com/JeffreyOlmo/Curvature_Subspace_Compression}
}
```

## References

- Merullo et al. (2025). "From Memorization to Reasoning in the Spectrum of Loss Curvature." arXiv:2510.24256
- Martens & Grosse (2015). "Optimizing Neural Networks with Kronecker-factored Approximate Curvature." arXiv:1503.05671

