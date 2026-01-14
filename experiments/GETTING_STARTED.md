# Getting Started with SSVAE Benchmarking

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with probtorch, install it separately:
```bash
pip install probtorch
```

### Step 2: Run Quick Test

```bash
cd experiments
python quick_test.py
```

This will:
- Download MNIST dataset (if not already present)
- Train a small SSVAE for 10 epochs
- Verify all components work correctly

**Expected output**: Test accuracy ~0.85-0.95 after 10 epochs

### Step 3: Run Full Experiments

```bash
python label_robustness_experiment.py --dataset MNIST --num_epochs 50
```

For a faster test, or:

```bash
python label_robustness_experiment.py
```

For the full experiment suite (both datasets, all conditions).

---

## What Gets Run

### Label Sparsity Experiment
Tests with decreasing labeled data:
- 10% → 5% → 2% → 1% → 0.5% → 0.1%

### Label Noise Experiment  
Tests with increasing corruption:
- 0% → 5% → 10% → 15% → 20% → 25% → 30%

### Metrics Computed
For each configuration:
- **Classification Accuracy**: How well the model predicts labels
- **Beta-VAE Score**: Disentanglement quality (0-1, higher better)
- **Factor-VAE Score**: Latent separation quality (higher better)
- **MIG Score**: Mutual information gap (higher better)

---

## Results

Results are saved to `../results/`:

```
results/
├── MNIST_sparsity_results.json          # Raw data
├── MNIST_noise_results.json             # Raw data
├── MNIST_robustness_analysis.png        # Plots
├── FashionMNIST_sparsity_results.json
├── FashionMNIST_noise_results.json
└── FashionMNIST_robustness_analysis.png
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

| Configuration | CPU Time | GPU Time |
|--------------|----------|----------|
| Quick test (10 epochs) | ~3-5 min | ~1 min |
| Single experiment (100 epochs) | ~30-60 min | ~5-15 min |
| Sparsity analysis (6 runs) | ~3-6 hours | ~30-90 min |
| Noise analysis (7 runs) | ~3.5-7 hours | ~35-105 min |
| **Full suite (both datasets)** | **~13-26 hours** | **~2-6 hours** |

**Recommendation**: Start with quick test and single dataset!

---

## Example Workflows

### Workflow 1: Quick Exploration
```bash
# 1. Test implementation
python quick_test.py

# 2. Run on MNIST only (faster)
python label_robustness_experiment.py --dataset MNIST --num_epochs 50

# 3. Analyze results
jupyter notebook analyze_results.ipynb
```
**Time**: ~1-2 hours (CPU) or ~15-30 min (GPU)

### Workflow 2: Sparsity Analysis Only
```bash
python label_robustness_experiment.py --sparsity_only --num_epochs 75
```
**Focus**: How much labeled data is needed?

### Workflow 3: Noise Analysis Only
```bash
python label_robustness_experiment.py --noise_only --label_fraction 0.1
```
**Focus**: How much label corruption can be tolerated?

### Workflow 4: Full Analysis
```bash
# Run overnight or on cluster
python label_robustness_experiment.py --num_epochs 100
```
**Result**: Complete robustness characterization

---

## Customization Examples

### Smaller Model (Faster)
```bash
python label_robustness_experiment.py \
    --num_style 32 \
    --num_epochs 50 \
    --dataset MNIST
```

### More Thorough Training
```bash
python label_robustness_experiment.py \
    --num_epochs 200 \
    --dataset FashionMNIST
```

### Specific Label Conditions
Edit `label_robustness_experiment.py`:

```python
# Line ~26-27, modify:
'label_fractions': [0.2, 0.1, 0.05, 0.01],  # Custom fractions
'corruption_rates': [0.0, 0.1, 0.2, 0.3],    # Custom corruptions
```

---

## Command-Line Options

```
--dataset {MNIST,FashionMNIST}    Choose dataset
--num_epochs INT                  Training epochs per run
--num_style INT                   Latent dimensions
--cuda {true,false}               Force GPU on/off
--label_fraction FLOAT            For noise experiment
--sparsity_only                   Skip noise experiment
--noise_only                      Skip sparsity experiment
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
# Install
pip install -r requirements.txt

# Test
python quick_test.py

# Run
python label_robustness_experiment.py --dataset MNIST --num_epochs 50

# Analyze
jupyter notebook analyze_results.ipynb
```

**Good luck with your experiments!** 🚀
