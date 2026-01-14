# 📊 SSVAE Label Robustness & Sparsity Benchmarking

Complete implementation of experiments to test the "break point" of Semi-Supervised VAE frameworks.

---

## 🎯 What This Does

Tests how SSVAE performance degrades when:
1. **Label Quantity ↓**: Reduce labeled data from 10% to 0.1%
2. **Label Quality ↓**: Corrupt 0-30% of labels to wrong classes

Measures:
- ✅ Classification accuracy
- ✅ Beta-VAE disentanglement score  
- ✅ Factor-VAE score
- ✅ Mutual Information Gap (MIG)

---

## 📁 What Was Created

```
experiments/
├── 📘 GETTING_STARTED.md              ← START HERE!
├── 📘 README.md                       ← Detailed documentation
├── 📘 IMPLEMENTATION_SUMMARY.md       ← Technical overview
│
├── 🐍 ssvae_model.py                  ← SSVAE architecture
├── 🐍 utils.py                        ← Training & data utilities
├── 🐍 metrics.py                      ← Disentanglement metrics
├── 🐍 label_robustness_experiment.py  ← MAIN EXPERIMENT SCRIPT
│
├── 🧪 quick_test.py                   ← Quick verification (10 epochs)
├── 📓 analyze_results.ipynb           ← Results visualization
├── 🔧 quick_start.sh                  ← Automated setup
└── 📦 requirements.txt                ← Dependencies
```

---

## 🚀 Quick Start (Copy-Paste)

```bash
# 1. Install dependencies
cd experiments
pip install -r requirements.txt

# 2. Quick test (verify everything works)
python quick_test.py

# 3. Run experiment (MNIST, 50 epochs - recommended for first run)
python label_robustness_experiment.py --dataset MNIST --num_epochs 50

# 4. Analyze results
jupyter notebook analyze_results.ipynb
```

**Estimated time**: 1-2 hours (CPU) or 15-30 minutes (GPU)

---

## 📊 Experiments Implemented

### Experiment 1: Label Sparsity Analysis
- **Varies**: Labeled data percentage (10% → 0.1%)
- **Fixed**: No label corruption (100% accurate labels)
- **Question**: How little labeled data can we use?

### Experiment 2: Label Noise Analysis  
- **Varies**: Label corruption rate (0% → 30%)
- **Fixed**: 10% labeled data
- **Question**: How much label noise can we tolerate?

---

## 📈 What You'll Get

### Automated Plots
- Accuracy vs Label Sparsity (log scale)
- Disentanglement vs Label Sparsity
- Accuracy vs Label Corruption
- Disentanglement vs Label Corruption
- Training curves over epochs
- Cross-dataset comparisons

### JSON Results
Complete metrics for every configuration:
```json
{
  "label_fraction": 0.1,
  "corruption_rate": 0.0,
  "final_test_accuracy": 0.952,
  "disentanglement_metrics": {
    "beta_vae": 0.823,
    "factor_vae": 2.145,
    "mig": 0.412
  }
}
```

---

## 🔬 Research Questions Answered

| Question | How to Find Answer |
|----------|-------------------|
| **At what % of labels does performance collapse?** | Check sparsity plots - look for sharp drop |
| **How much corruption is tolerable?** | Check noise plots - find degradation threshold |
| **Is disentanglement robust to label quality?** | Compare Beta-VAE vs accuracy slopes |
| **Does latent space collapse with bad labels?** | Check Beta-VAE at high corruption |
| **MNIST vs FashionMNIST differences?** | Compare plots between datasets |

---

## ⚙️ Customization Points

### Quick Changes (Command Line)
```bash
# Different dataset
--dataset FashionMNIST

# Longer training
--num_epochs 100

# Smaller model (faster)
--num_style 32

# Only one experiment type
--sparsity_only
--noise_only
```

### Code Changes
Edit `label_robustness_experiment.py`:

```python
DEFAULT_CONFIG = {
    # Line 27-28: Change test conditions
    'label_fractions': [0.1, 0.05, 0.02, 0.01],
    'corruption_rates': [0.0, 0.1, 0.2, 0.3],
    
    # Line 15-16: Change model size
    'num_hidden': 256,
    'num_style': 50,
}
```

---

## 💡 Implementation Highlights

### Architecture
- **Encoder**: Gumbel-Softmax for discrete labels + Gaussian for continuous latents
- **Decoder**: Reconstructs images from combined label + style representation
- **Loss**: ELBO with importance weighting

### Disentanglement Metrics

1. **Beta-VAE Score** (0-1, higher better)
   - Trains classifiers on single latent dimensions
   - Measures how exclusively each dimension encodes class info

2. **Factor-VAE Score** (higher better)
   - Ratio of between-class to within-class variance
   - Measures latent space separation quality

3. **MIG Score** (higher better)
   - Mutual information gap between top dimensions
   - Measures factor encoding exclusivity

### Key Features
- ✅ Monte Carlo sampling for stable estimates
- ✅ CUDA acceleration
- ✅ Systematic label corruption
- ✅ Flexible labeled data fractions
- ✅ Comprehensive metrics tracking
- ✅ Automated result saving & plotting

---

## ⏱️ Time Estimates

| What | CPU | GPU |
|------|-----|-----|
| Quick test | 3-5 min | 1 min |
| Single MNIST run | 30-60 min | 5-15 min |
| Sparsity analysis (6 configs) | 3-6 hrs | 30-90 min |
| Noise analysis (7 configs) | 3.5-7 hrs | 35-105 min |
| **Full suite (both datasets)** | **13-26 hrs** | **2-6 hrs** |

**💡 Tip**: Start with MNIST only and 50 epochs!

---

## 📚 Files Guide

| File | Purpose | When to Use |
|------|---------|-------------|
| `GETTING_STARTED.md` | Step-by-step tutorial | First time setup |
| `README.md` | Detailed documentation | Understanding experiment design |
| `IMPLEMENTATION_SUMMARY.md` | Technical details | Understanding implementation |
| `quick_test.py` | Verification script | Before running full experiments |
| `label_robustness_experiment.py` | Main experiment | Running benchmarks |
| `analyze_results.ipynb` | Results analysis | After experiments complete |

---

## 🎓 Based On

**Paper**: Kingma et al. (2014) - "Semi-supervised learning with deep generative models"

**Framework**: probtorch - Probabilistic programming and deep learning

**Datasets**: MNIST & FashionMNIST

---

## ✅ Verification Checklist

- [x] SSVAE model implementation
- [x] Label corruption utilities
- [x] Disentanglement metrics (Beta-VAE, Factor-VAE, MIG)
- [x] Label sparsity experiment
- [x] Label noise experiment  
- [x] Automated result saving
- [x] Plotting & visualization
- [x] Analysis notebook
- [x] Documentation
- [x] Quick test script
- [x] Example workflows

---

## 🚦 Next Steps

1. **Read** `GETTING_STARTED.md` for setup instructions
2. **Run** `quick_test.py` to verify installation
3. **Execute** experiment with your chosen parameters
4. **Analyze** results in Jupyter notebook
5. **Report** your findings!

---

## 🆘 Common Issues

**"No module named probtorch"** → `pip install probtorch`

**"CUDA out of memory"** → `--cuda false` or reduce batch size

**"Takes too long"** → `--num_epochs 30` or `--dataset MNIST` only

**"No results found"** → Run experiments first to generate data

---

## 📞 Support

- Check existing examples in `../examples/` directory
- Review `README.md` for troubleshooting
- Consult probtorch documentation

---

**Ready to benchmark? Start with:**
```bash
python quick_test.py
```

**Good luck! 🎉**
