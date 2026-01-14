# SSVAE Label Robustness and Sparsity Analysis

This folder contains experiments to evaluate the robustness of Semi-Supervised Variational Autoencoders (SSVAE) to label quality and quantity.

## Overview

The experiments test the "break point" of the SSVAE framework by systematically varying:

1. **Label Quantity**: Reduce the percentage of labeled data (from 10% down to 0.1%)
2. **Label Noise**: Introduce corrupted labels (flip 10-30% of labels to wrong classes)

The goal is to measure how sensitive disentanglement quality is to supervision quality, specifically:
- Does the latent space collapse when 20% of labels are wrong?
- How much labeled data is needed to maintain good disentanglement?
- What is the relationship between classification accuracy and disentanglement metrics?

## Files

- `ssvae_model.py`: SSVAE encoder/decoder architecture using probtorch
- `utils.py`: Utilities for data loading, label corruption, and training
- `metrics.py`: Implementation of disentanglement metrics (Beta-VAE, Factor-VAE, MIG)
- `label_robustness_experiment.py`: Main experiment script
- `README.md`: This file

## Installation

Ensure you have the required dependencies:

```bash
pip install torch torchvision probtorch numpy scipy scikit-learn matplotlib seaborn
```

## Usage

### Quick Start

Run experiments on both MNIST and FashionMNIST:

```bash
python label_robustness_experiment.py
```

### Run on Specific Dataset

```bash
# MNIST only
python label_robustness_experiment.py --dataset MNIST

# FashionMNIST only
python label_robustness_experiment.py --dataset FashionMNIST
```

### Customize Training

```bash
# Shorter training for quick testing
python label_robustness_experiment.py --num_epochs 50 --dataset MNIST

# Change latent dimensions
python label_robustness_experiment.py --num_style 32

# Force CPU (no CUDA)
python label_robustness_experiment.py --cuda false
```

### Run Specific Experiments

```bash
# Only label sparsity experiment
python label_robustness_experiment.py --sparsity_only

# Only label noise experiment
python label_robustness_experiment.py --noise_only --label_fraction 0.05
```

## Experiments

### 1. Label Sparsity Analysis

Trains models with varying amounts of labeled data (10%, 5%, 2%, 1%, 0.5%, 0.1%) with no label corruption.

**Metrics Measured:**
- Classification accuracy
- Beta-VAE disentanglement score
- Factor-VAE score
- Mutual Information Gap (MIG)

**Expected Findings:**
- Classification accuracy should decrease as labeled data decreases
- Disentanglement quality may remain stable until a critical threshold
- Identification of the minimum labeled data needed for good representations

### 2. Label Noise Analysis

Trains models with fixed labeled data (default 10%) but varying corruption rates (0%, 5%, 10%, 15%, 20%, 25%, 30%).

**Metrics Measured:**
- Classification accuracy
- Beta-VAE disentanglement score
- Factor-VAE score
- Mutual Information Gap (MIG)

**Expected Findings:**
- Both accuracy and disentanglement should degrade with more noise
- Potential non-linear relationship between corruption and performance
- Identification of corruption tolerance threshold

## Results

Results are saved to `../results/` directory:

- `{dataset}_sparsity_results.json`: Label sparsity experiment results
- `{dataset}_noise_results.json`: Label noise experiment results
- `{dataset}_robustness_analysis.png`: Visualization plots

### Result Format

Each JSON file contains a list of experiment runs with:

```json
{
  "dataset": "MNIST",
  "label_fraction": 0.1,
  "corruption_rate": 0.0,
  "final_test_accuracy": 0.952,
  "final_test_elbo": 1234.5,
  "disentanglement_metrics": {
    "beta_vae": 0.823,
    "factor_vae": 2.145,
    "mig": 0.412
  },
  "train_elbos": [...],
  "test_elbos": [...],
  "test_accuracies": [...]
}
```

## Disentanglement Metrics

### Beta-VAE Score

Measures how well individual latent dimensions correspond to class labels. A classifier is trained to predict class labels using single latent dimensions. Higher scores (closer to 1.0) indicate better disentanglement.

### Factor-VAE Score

Measures the ratio of between-class variance to within-class variance for each latent dimension. Higher scores indicate latent dimensions that clearly separate different classes.

### Mutual Information Gap (MIG)

Measures the difference between the top two latent dimensions in terms of mutual information with class labels. Quantifies how exclusively each factor is encoded.

## Interpretation Guide

### Key Questions to Answer

1. **At what label fraction does performance collapse?**
   - Look for sharp drops in accuracy and disentanglement metrics

2. **How much noise can the model tolerate?**
   - Identify the corruption rate where metrics significantly degrade

3. **Is disentanglement more robust than accuracy?**
   - Compare relative changes in accuracy vs disentanglement metrics

4. **Does the relationship differ between datasets?**
   - Compare MNIST (simpler) vs FashionMNIST (more complex)

## Customization

### Modify Label Fractions

Edit `DEFAULT_CONFIG` in `label_robustness_experiment.py`:

```python
'label_fractions': [0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
```

### Modify Corruption Rates

```python
'corruption_rates': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
```

### Change Model Architecture

```python
'num_hidden': 256,  # Hidden layer size
'num_style': 50,    # Latent style dimensions
```

## Computational Requirements

- **Time**: Each experiment run takes ~5-15 minutes on GPU, ~30-60 minutes on CPU
- **Memory**: ~2GB GPU memory, ~4GB RAM
- **Storage**: Results are small (~100KB per experiment)

## Citation

If you use this code, please cite the original SSVAE paper:

```
Kingma, D. P., Rezende, D. J., Mohamed, S., & Welling, M. (2014).
Semi-supervised learning with deep generative models.
Advances in neural information processing systems, 27.
```

## Troubleshooting

### Out of Memory Errors

Reduce batch size or number of style dimensions:

```bash
python label_robustness_experiment.py --num_style 32
```

Edit `DEFAULT_CONFIG['num_batch']` to reduce batch size.

### Slow Training

Use CUDA if available or reduce epochs:

```bash
python label_robustness_experiment.py --num_epochs 50 --cuda true
```

### Import Errors

Ensure all dependencies are installed and you're running from the correct directory:

```bash
cd /path/to/ssvae/experiments
python label_robustness_experiment.py
```
