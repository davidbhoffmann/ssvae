# Combined 3D Parameter Sweep Experiment

## Overview

The combined robustness experiment explores the interaction between three critical hyperparameters:
- **Label Fraction**: Amount of labeled data (10% → 0.1%)
- **Label Corruption**: Noise in labels (0% → 30%)
- **Alpha Parameter**: ELBO weighting factor (0.01 → 1.0)

## Alpha Parameter

The **alpha** parameter controls the weighting in the ELBO (Evidence Lower Bound) objective function:

```python
ELBO = E_q[log p(x,y,z)] - alpha * KL(q(z|x,y) || p(z))
```

### Effect of Alpha Values:

- **α < 0.1**: Emphasizes reconstruction quality, may lead to posterior collapse
- **α = 0.1** (default): Balanced trade-off between reconstruction and regularization
- **α > 0.5**: Stronger KL regularization, promotes better disentanglement
- **α → 1.0**: Standard VAE objective, maximum regularization

### Typical Ranges:
- Conservative: [0.1, 0.2, 0.5]
- Broad exploration: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
- Default experiment: [0.1, 0.5, 1.0]

## Running the Experiment

### Basic Usage

```bash
# Default: 3×3×3 = 27 experiments (MNIST + FashionMNIST = 54 total)
conda run -n vae python combined_robustness_experiment.py

# Custom parameters
conda run -n vae python combined_robustness_experiment.py \
    --dataset MNIST \
    --num_epochs 100 \
    --label_fractions 0.1,0.05,0.02 \
    --corruption_rates 0.0,0.1,0.2 \
    --alpha_values 0.1,0.5,1.0
```

### Quick Test (Reduced Parameters)

```bash
# Fast test: 2×2×2 = 8 experiments, 5 epochs each
conda run -n vae python combined_robustness_experiment.py \
    --dataset MNIST \
    --num_epochs 5 \
    --label_fractions 0.1,0.05 \
    --corruption_rates 0.0,0.1 \
    --alpha_values 0.1,0.5
```

## Output Files

Results are saved in:
```
results/
  combined_MNIST_results.json      # Raw experimental data
  combined_MNIST_results.png       # Visualization plots
  combined_FashionMNIST_results.json
  combined_FashionMNIST_results.png
```

## Visualizations

The experiment generates:

1. **Heatmaps (one per alpha value)**:
   - X-axis: Label fraction
   - Y-axis: Corruption rate
   - Color: Metric value (accuracy, Beta-VAE, Factor-VAE, MIG)

2. **Interaction Plots**:
   - Shows how alpha affects the relationship between labels/noise and performance
   - Helps identify optimal alpha for different data conditions

## Interpreting Results

### Key Questions Addressed:

1. **Does alpha affect label efficiency?**
   - Compare accuracy heatmaps across different alpha values
   - Lower alpha might work better with sparse labels

2. **Does alpha help with label noise?**
   - Check if higher alpha provides more robust representations
   - Compare noise tolerance across alpha values

3. **What's the optimal alpha for disentanglement?**
   - Higher alpha generally improves Beta-VAE and Factor-VAE scores
   - But may hurt classification accuracy

4. **Are there interaction effects?**
   - Does optimal alpha change with label fraction?
   - Does optimal alpha change with corruption rate?

## Expected Runtime

- **Quick test** (2×2×2, 5 epochs): ~10-15 minutes on MPS/CUDA
- **Small sweep** (3×3×3, 50 epochs): ~2-4 hours on MPS/CUDA
- **Full sweep** (6×7×6, 100 epochs): ~24-48 hours on MPS/CUDA

## Implementation Notes

### Code Updates Made:

1. **utils.py**: Added `alpha` parameter to `train_epoch()` and `test_epoch()`
2. **metrics.py**: Updated all metrics functions to use `device` parameter
3. **ssvae_model.py**: ELBO function already supported alpha parameter

### Device Support:

The experiment automatically detects and uses:
1. CUDA (NVIDIA GPUs) - if available
2. MPS (Apple Silicon) - if available
3. CPU - fallback

### Compatibility:

- Python 3.10+ (with collections.MutableMapping patch)
- PyTorch with CUDA/MPS support
- probtorch 0.4+

## Example Analysis Workflow

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('results/combined_MNIST_results.json', 'r') as f:
    results = json.load(f)

# Find best alpha for each condition
for label_frac in results['label_fractions']:
    for corruption in results['corruption_rates']:
        # Get accuracies for all alphas
        accs = []
        for alpha in results['alpha_values']:
            key = f"{label_frac}_{corruption}_{alpha}"
            accs.append(results['results'][key]['accuracy'])
        
        best_alpha = results['alpha_values'][np.argmax(accs)]
        print(f"Labels: {label_frac*100}%, Noise: {corruption*100}%, "
              f"Best α: {best_alpha}, Acc: {max(accs):.3f}")
```

## Related Experiments

- **label_robustness_experiment.py**: Separate sweeps over labels and noise (fixed α=0.1)
- **quick_test.py**: Fast verification (10 epochs, α=0.1)

## References

- Original SSVAE paper: Kingma et al., "Semi-supervised Learning with Deep Generative Models" (2014)
- Beta-VAE: Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)
- Factor-VAE: Kim & Mnih, "Disentangling by Factorising" (2018)
