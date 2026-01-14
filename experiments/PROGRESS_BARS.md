# Progress Bar Documentation

## Overview

All experiment scripts now include **tqdm progress bars** to provide real-time feedback on:
- Overall experiment progress
- Individual epoch training progress  
- Current performance metrics

## Installation

```bash
conda run -n vae pip install tqdm
```

Or it's automatically installed via:
```bash
pip install -r requirements.txt
```

## Progress Bar Features

### 1. Quick Test (`quick_test.py`)

Shows a single progress bar for 10 epochs:

```
Quick Test: 100%|██████████| 10/10 [00:15<00:00, 1.54s/it, Train ELBO=8.28e-01, Test Acc=0.721, Test ELBO=8.21e-01]
```

**Displays:**
- Progress percentage and bar
- Epoch counter (e.g., 10/10)
- Time elapsed and estimated time remaining
- Real-time metrics: Train ELBO, Test Accuracy, Test ELBO

### 2. Label Robustness Experiment (`label_robustness_experiment.py`)

Shows two levels of progress:

**Outer Loop (Experiments):**
```
MNIST Label Sparsity: 100%|██████████| 6/6 [2:30:00<00:00, 1500.0s/it]
```

**Inner Loop (Epochs per experiment):**
```
Labels:10.0% Noise:0.0%: 100%|██████████| 100/100 [00:25<00:00, 3.96it/s, Train ELBO=8.15e-01, Test Acc=0.745, Test ELBO=8.10e-01]
```

**Displays:**
- Experiment parameters (label fraction, noise level)
- Epoch progress within each experiment
- Live training metrics

### 3. Combined Experiment (`combined_robustness_experiment.py`)

Shows **nested progress bars** for 3D parameter sweep:

**Outer Loop (Parameter combinations):**
```
MNIST Experiments | Labels:10.0% Noise:0% α:0.10: 12/27 [01:30:00<02:00:00, 480.0s/it]
```

**Inner Loop (Epochs per combination):**
```
Training (α=0.10): 100%|██████████| 75/75 [00:18<00:00, 4.12it/s, Train ELBO=8.20e-01, Test Acc=0.732, Test ELBO=8.15e-01]
```

**Displays:**
- Current parameter combination (labels, noise, alpha)
- Overall progress through all combinations
- Individual training progress per combination
- Real-time performance metrics

## Benefits

### 1. **Visibility**
- Immediately see if the experiment is running
- No need to check log files or wait for periodic prints

### 2. **Time Estimation**
- See how long each epoch takes
- Estimate total remaining time
- Plan when to check results

### 3. **Performance Monitoring**
- Watch metrics update in real-time
- Spot issues early (e.g., NaN losses, stuck training)
- Verify model is learning

### 4. **Multi-level Tracking**
For combined experiments with many parameter combinations:
- Track position in overall sweep
- See progress within current experiment
- Know exactly which configuration is running

## Example Output

### Running Combined Experiment

```bash
conda run -n vae python combined_robustness_experiment.py \
    --dataset MNIST \
    --num_epochs 50 \
    --label_fractions 0.1,0.05,0.02 \
    --corruption_rates 0.0,0.1,0.2 \
    --alpha_values 0.1,0.5,1.0
```

**Output:**
```
================================================================================
STARTING COMBINED SWEEP FOR MNIST
================================================================================

################################################################################
COMBINED 3D PARAMETER SWEEP - MNIST
################################################################################

Parameter Space:
  Label Fractions: [0.1, 0.05, 0.02]
  Corruption Rates: [0.0, 0.1, 0.2]
  Alpha Values: [0.1, 0.5, 1.0]
  Total Experiments: 27

MNIST | Labels:10.0% Noise:0% α:0.10:  4%|▍  | 1/27 [00:45<19:35, 45.2s/it]
Training (α=0.10):  20%|██  | 10/50 [00:02<00:10, 3.89it/s, Train ELBO=9.45e-01, Test Acc=0.524, Test ELBO=9.12e-01]
```

### Completed Experiment

```
MNIST | Labels:2.0% Noise:20% α:1.00: 100%|██████████| 27/27 [20:15<00:00, 45.0s/it]

Computing disentanglement metrics...
Saving results to results/combined_MNIST_results.json
Generating plots...
Saving plots to results/combined_MNIST_results.png

✓ Experiment completed successfully!
```

## Technical Details

### Configuration

Progress bars use `tqdm` with:
- `desc`: Description of current task
- `position`: Nesting level (0 = outer, 1 = inner)
- `leave`: Whether to keep bar after completion
- `set_postfix()`: Update real-time metrics

### Performance Impact

Minimal overhead:
- ~0.1% slowdown per progress bar
- Negligible for deep learning experiments
- Much faster than periodic print statements

### Terminal Compatibility

Works with:
- ✅ Standard terminals (bash, zsh, fish)
- ✅ Jupyter notebooks (auto-detects and uses widget)
- ✅ VS Code integrated terminal
- ✅ tmux/screen sessions
- ✅ SSH sessions

## Customization

You can adjust progress bar behavior by modifying:

```python
# In experiment scripts
pbar = tqdm(
    range(num_epochs), 
    desc="Training",
    position=0,          # Nesting level
    leave=True,          # Keep bar after completion
    ncols=100,          # Width in characters
    unit="epoch",       # Unit name
    colour="green"      # Bar color
)
```

## Troubleshooting

### Progress bars not showing

Check if running in background mode:
```bash
# This works (foreground)
python experiment.py

# This might not show progress (background)
python experiment.py &
```

### Jumbled output

If using multiple threads/processes, disable nested bars:
```python
pbar = tqdm(..., position=0, leave=True)
```

### Want to disable progress bars

Set environment variable:
```bash
export TQDM_DISABLE=1
python experiment.py
```

Or modify code:
```python
from tqdm import tqdm
tqdm = lambda x, **kwargs: x  # Disable all tqdm bars
```

## See Also

- [tqdm Documentation](https://github.com/tqdm/tqdm)
- [Quick Test Script](quick_test.py) - Simple example
- [Combined Experiment](combined_robustness_experiment.py) - Nested bars
