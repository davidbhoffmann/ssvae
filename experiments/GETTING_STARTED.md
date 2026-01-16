# Getting Started with SSVAE Benchmarking

## Quick Start (3 Steps)

### Step 1: Setup Environment

**If using conda (recommended):**
```bash
# Activate your environment
conda activate vae

# Install dependencies
pip install -r requirements.txt
```

**Or using pip directly:**
```bash
pip install -r requirements.txt
```

**Note**: All dependencies including `tqdm` (for progress bars) will be installed automatically.

### Step 2: Run Quick Test

```bash
conda run -n vae python quick_test.py
# Or if not using conda: python quick_test.py
```

This will:
- Download MNIST dataset (if not already present)
- Train a small SSVAE for 10 epochs
- Show **live progress bar** with training metrics
- Verify all components work correctly

**Expected output**: 
```
Quick Test: 100%|██████████| 10/10 [00:15<00:00, Train ELBO=8.28e-01, Test Acc=0.721, ...]
✓ Test completed successfully!
Final Results:
  Test Accuracy: 0.721
```

### Step 3: Run Experiments

You have three experiment options:

**A) Label Robustness (Separate Sparsity & Noise):**
```bash
conda run -n vae python label_robustness_experiment.py --dataset MNIST --num_epochs 50
```
Tests label sparsity and noise separately (~1-2 hours on MPS/CUDA).

**B) Combined 3D Sweep (Labels × Noise × Alpha):**
```bash
conda run -n vae 

```
Explores full 3D parameter space (3×3×3 = 27 experiments ~2-4 hours).

**C) Quick Combined Test:**
```bash
conda run -n vae python combined_robustness_experiment.py \
    --dataset MNIST --num_epochs 5 \
    --label_fractions 0.1,0.05 \
    --corruption_rates 0.0,0.1 \
    --alpha_values 0.1,0.5
```

```bash
python combined_robustness_experiment.py \
    --dataset MNIST --num_epochs 75 \
    --n_labels 10,50,100,600,1000,3000,10000 \
    --corruption_rates 0.0,0.01,0.05,0.1,0.2,0.4,0.8 \
    --alpha_values 50
```
Fast test of combined experiment (2×2×2 = 8 experiments, ~10-15 min).

---

## What Gets Run

### Experiment 1: Label Robustness (`label_robustness_experiment.py`)

**Label Sparsity:** Tests with decreasing labeled data:
- 10% → 5% → 2% → 1% → 0.5% → 0.1%

**Label Noise:** Tests with increasing corruption:
- 0% → 5% → 10% → 15% → 20% → 25% → 30%

### Experiment 2: Combined 3D Sweep (`combined_robustness_experiment.py`)

Explores interactions between three factors:
- **Label Fraction**: Amount of labeled data (default: 10%, 5%, 2%)
- **Corruption Rate**: Label noise level (default: 0%, 10%, 20%)
- **Alpha Parameter**: ELBO weighting (default: 0.1, 0.5, 1.0)

Total combinations: 3 × 3 × 3 = **27 experiments per dataset**

### Metrics Computed
For each configuration:
- **Classification Accuracy**: How well the model predicts labels
- **Beta-VAE Score**: Disentanglement quality (0-1, higher better)
- **Factor-VAE Score**: Latent separation quality (higher better)
- **MIG Score**: Mutual information gap (higher better)

---

## Results

Results are saved to `results/` directory:

**Label Robustness Experiment:**
```
results/
├── MNIST_sparsity_results.json          # Raw data (label sparsity)
├── MNIST_noise_results.json             # Raw data (label noise)
├── MNIST_robustness_analysis.png        # 6-panel plot
├── FashionMNIST_sparsity_results.json
├── FashionMNIST_noise_results.json
└── FashionMNIST_robustness_analysis.png
```

**Combined Experiment:**
```
results/
├── combined_MNIST_results.json          # 3D sweep raw data
├── combined_MNIST_results.png           # Heatmaps + interaction plots
├── combined_FashionMNIST_results.json
└── combined_FashionMNIST_results.png
```

### Analyze Results

```bash
jupyter notebook analyze_results.ipynb
```

This notebook will:
- Load all results
- Create comparison plots
- Generate summary statistics
- Help answer key research questions

---

## Time Estimates

**Device Auto-Detection:** All scripts automatically use CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.

| Configuration | CPU | MPS/CUDA |
|--------------|-----|----------|
| Quick test (10 epochs) | ~3-5 min | ~30-60 sec |
| Single experiment (100 epochs) | ~30-60 min | ~5-15 min |
| Label sparsity (6 runs × 100 epochs) | ~3-6 hours | ~30-90 min |
| Label noise (7 runs × 100 epochs) | ~3.5-7 hours | ~35-105 min |
| **Combined 3D (27 runs × 75 epochs)** | **~8-16 hours** | **~2-4 hours** |
| Full suite (both datasets) | ~26-52 hours | ~4-10 hours |

**Recommendations**: 
- Start with `quick_test.py` (30-60 seconds)
- Then try quick combined test (10-15 minutes)
- Use single dataset before running both

---

## Live Progress Tracking

All experiments now show **real-time progress bars** with live metrics:

```
MNIST | Labels:10.0% Noise:0% α:0.10: 8/27 [06:00<13:30, 42.6s/it]
└─ Training (α=0.10): 65%|██████▌| 32/50 [00:08<00:04, Train ELBO=8.45e-01, Test Acc=0.685]
```

**What you see:**
- Overall experiment progress (e.g., 8/27 combinations)
- Current parameter configuration
- Epoch progress within current experiment
- Live training metrics (ELBO, accuracy)
- Time elapsed and estimated time remaining

See [PROGRESS_BARS.md](PROGRESS_BARS.md) for details.

---

## Example Workflows

### Workflow 1: Quick Exploration (Recommended)
```bash
# 1. Test implementation (~30-60 sec)
conda run -n vae python quick_test.py

# 2. Quick combined test (~10-15 min)
conda run -n vae python combined_robustness_experiment.py \
    --dataset MNIST --num_epochs 5 \
    --label_fractions 0.1,0.05 \
    --corruption_rates 0.0,0.1 \
    --alpha_values 0.1,0.5

# 3. Analyze results
jupyter notebook analyze_results.ipynb
```
**Time**: ~15-20 minutes total (MPS/CUDA) or ~45-60 min (CPU)

### Workflow 2: Full 3D Parameter Sweep
```bash
conda run -n vae python combined_robustness_experiment.py \
    --dataset MNIST --num_epochs 75
```
**Focus**: Understand interactions between labels, noise, and alpha.  
**Time**: ~2-4 hours (MPS/CUDA)

### Workflow 3: Label Robustness Only
```bash
conda run -n vae python label_robustness_experiment.py \
    --dataset MNIST --num_epochs 100
```
**Focus**: Traditional sparsity and noise analysis (separate).  
**Time**: ~1-2 hours (MPS/CUDA)

### Workflow 4: Complete Suite (All Datasets)
```bash
# Run overnight or on cluster
conda run -n vae python combined_robustness_experiment.py --num_epochs 100
# Then:
conda run -n vae python label_robustness_experiment.py --num_epochs 100
```
**Result**: Full characterization with both experiment types.  
**Time**: ~6-14 hours (MPS/CUDA)

---

## Customization Examples

### Custom Alpha Values (Combined Experiment)
```bash
conda run -n vae python combined_robustness_experiment.py \
    --dataset MNIST --num_epochs 50 \
    --alpha_values 0.01,0.05,0.1,0.2,0.5,1.0
```
Test wider range of ELBO weightings (6 alphas × 3 labels × 3 noise = 54 experiments).

### Dense Parameter Grid
```bash
conda run -n vae python combined_robustness_experiment.py \
    --dataset MNIST --num_epochs 50 \
    --label_fractions 0.1,0.08,0.05,0.03,0.02,0.01 \
    --corruption_rates 0.0,0.05,0.1,0.15,0.2,0.25,0.3 \
    --alpha_values 0.1,0.3,0.5,0.7,1.0
```
Fine-grained sweep (6 × 7 × 5 = 210 experiments, ~15-30 hours MPS/CUDA).

### Specific Device Selection
```bash
# Force CPU (useful for debugging)
conda run -n vae python combined_robustness_experiment.py --device cpu

# Force CUDA
conda run -n vae python combined_robustness_experiment.py --device cuda

# Force MPS (Apple Silicon)
conda run -n vae python combined_robustness_experiment.py --device mps
```

### Faster Testing
```bash
# Smaller model, fewer epochs
conda run -n vae python combined_robustness_experiment.py \
    --num_style 32 \
    --num_epochs 25 \
    --dataset MNIST
```

---

## Command-Line Options

### All Experiments
```
--dataset {MNIST,FashionMNIST}    Choose dataset (default: both)
--num_epochs INT                  Training epochs per experiment (default: 100)
--num_style INT                   Latent dimensions (default: 50)
--device {auto,cuda,mps,cpu}      Force specific device (default: auto)
```

### Label Robustness Experiment Only
```
--label_fraction FLOAT            For noise experiment (default: 0.1)
--sparsity_only                   Skip noise experiment
--noise_only                      Skip sparsity experiment
```

### Combined Experiment Only
```
--label_fractions STR             Comma-separated list (default: 0.1,0.05,0.02)
--corruption_rates STR            Comma-separated list (default: 0.0,0.1,0.2)
--alpha_values STR                Comma-separated list (default: 0.1,0.5,1.0)
```

**Examples:**
```bash
# Single dataset, 50 epochs
conda run -n vae python combined_robustness_experiment.py \
    --dataset MNIST --num_epochs 50

# Custom parameter ranges
conda run -n vae python combined_robustness_experiment.py \
    --label_fractions 0.2,0.1,0.05 \
    --corruption_rates 0.0,0.15,0.3 \
    --alpha_values 0.05,0.1,0.5
```

---

## Interpreting Results

### Key Questions

1. **At what label % does accuracy drop sharply?**
   - Look at sparsity plots
   - Typically around 1-2% for MNIST

2. **When does disentanglement collapse?**
   - Compare Beta-VAE scores across conditions
   - May be more robust than accuracy

3. **Corruption tolerance?**
   - Check noise plots
   - Expect degradation around 20-30%

4. **Is disentanglement robust?**
   - Compare slopes: accuracy vs Beta-VAE
   - Steeper slope = more sensitive

### Good Results Look Like

**Label Sparsity**:
- Accuracy: Gradual decrease, sharp drop at some threshold
- Disentanglement: More stable, slower decrease

**Label Noise**:
- Accuracy: Linear or slightly convex decrease
- Disentanglement: May show non-linear degradation

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'probtorch'"
```bash
pip install probtorch
```

### "CUDA out of memory"
```bash
python label_robustness_experiment.py --cuda false
# Or reduce batch size in code (num_batch = 64)
```

### "Experiment takes too long"
```bash
# Reduce epochs
python label_robustness_experiment.py --num_epochs 30

# Or use GPU
python label_robustness_experiment.py --cuda true
```

### "No results found" in notebook
First run experiments to generate results:
```bash
python label_robustness_experiment.py --dataset MNIST
```

---

## Next Steps After Running

1. **Analyze Results**: Use the notebook to explore findings
2. **Write Report**: Summarize key insights
3. **Compare Literature**: How do results match expectations?
4. **Extend**: Try different architectures, datasets, or metrics

---

## Support

- **Examples**: See `../examples/` directory
- **Documentation**: See `README.md` for details
- **Implementation**: See `IMPLEMENTATION_SUMMARY.md`

## Ready to Start!

```bash
# 1. Setup (one time)
conda activate vae
pip install -r requirements.txt

# 2. Quick test (~30-60 seconds)
conda run -n vae python quick_test.py

# 3. Run experiments (choose one):

# Option A: Quick combined test (~10-15 min)
conda run -n vae python combined_robustness_experiment.py \
    --dataset MNIST --num_epochs 5 \
    --label_fractions 0.1,0.05 --corruption_rates 0.0,0.1 --alpha_values 0.1,0.5

# Option B: Full 3D sweep (~2-4 hours)
conda run -n vae python combined_robustness_experiment.py --dataset MNIST

# Option C: Traditional robustness (~1-2 hours)
conda run -n vae python label_robustness_experiment.py --dataset MNIST

# 4. Analyze
jupyter notebook analyze_results.ipynb
```

**All experiments show live progress bars!** 📊  
**Good luck with your experiments!** 🚀
