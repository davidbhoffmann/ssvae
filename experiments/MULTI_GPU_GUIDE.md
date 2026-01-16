# Multi-GPU Support and Performance Optimization Guide

This guide explains the new multi-GPU support and performance optimizations added to the SSVAE experiments.

## Overview of Improvements

### 1. Multi-GPU Support
- **Automatic detection**: The code automatically detects and uses all available GPUs
- **Manual control**: You can specify which GPUs to use with `--gpu_ids`
- **Easy toggling**: Use `--no_multi_gpu` to disable multi-GPU even when available
- **DataParallel**: Uses PyTorch's DataParallel for efficient multi-GPU training

### 2. Data Loading Optimization
- **Parallel data loading**: Increased default workers from 0 to 4
- **Pinned memory**: Enabled for faster GPU transfer
- **Persistent workers**: Reuses worker processes across epochs

### 3. Evaluation Optimization
- **Configurable frequency**: Evaluate every N epochs with `--eval_frequency`
- **Faster training**: Skip evaluation on intermediate epochs to speed up long runs

## Usage Examples

### Basic Multi-GPU Training

Run an experiment with automatic GPU detection:
```bash
python combined_robustness_experiment.py --dataset MNIST --num_epochs 75
```

### Specify GPU IDs

Use specific GPUs (e.g., GPUs 0, 1, and 2):
```bash
python combined_robustness_experiment.py \
    --dataset MNIST \
    --num_epochs 75 \
    --gpu_ids 0,1,2
```

### Disable Multi-GPU

Force single GPU or CPU usage:
```bash
python combined_robustness_experiment.py \
    --dataset MNIST \
    --num_epochs 75 \
    --no_multi_gpu
```

### Optimize Data Loading

Increase data loading workers for faster throughput:
```bash
python combined_robustness_experiment.py \
    --dataset MNIST \
    --num_epochs 75 \
    --num_workers 8
```

### Speed Up Long Runs

Evaluate less frequently to speed up training:
```bash
python combined_robustness_experiment.py \
    --dataset MNIST \
    --num_epochs 75 \
    --eval_frequency 5  # Evaluate every 5 epochs instead of every epoch
```

### Combined Optimizations

All optimizations together:
```bash
python combined_robustness_experiment.py \
    --dataset MNIST \
    --num_epochs 75 \
    --gpu_ids 0,1,2,3 \
    --num_workers 8 \
    --eval_frequency 5
```

## SLURM Usage

### Multi-GPU SLURM Script

A new SLURM script `submit_multi_gpu.sh` is provided for multi-GPU jobs:

```bash
sbatch submit_multi_gpu.sh
```

### Customize SLURM Resources

Edit the SLURM script to request specific resources:

```bash
#SBATCH --gres=gpu:4          # Request 4 GPUs
#SBATCH --cpus-per-task=16    # 16 CPUs (4 per GPU is a good rule)
#SBATCH --mem=64G             # 64GB RAM
```

### SLURM Array Jobs

For running multiple configurations in parallel, use SLURM array jobs:

```bash
#!/bin/bash
#SBATCH --array=0-9           # Run 10 jobs in parallel
#SBATCH --gres=gpu:2          # 2 GPUs per job

# Each job runs different seed
python combined_robustness_experiment.py \
    --dataset MNIST \
    --num_seeds 1 \
    --seed_offset $SLURM_ARRAY_TASK_ID
```

## Command-Line Arguments

### New Arguments

All experiment scripts now support:

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_workers` | 4 | Number of parallel data loading workers |
| `--no_multi_gpu` | False | Disable multi-GPU training |
| `--gpu_ids` | None | Comma-separated GPU IDs (e.g., "0,1,2,3") |
| `--eval_frequency` | 1 | Evaluate every N epochs |

### Existing Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | both | MNIST or FashionMNIST |
| `--num_epochs` | 75/100 | Number of training epochs |
| `--num_style` | 10/50 | Number of latent style dimensions |
| `--device` | auto | Device to use (auto/cuda/mps/cpu) |

## Performance Tips

### 1. Optimal Number of Workers

- **Rule of thumb**: 4 workers per GPU is often optimal
- **Too few**: Underutilizes GPUs (they wait for data)
- **Too many**: Overhead from process management
- **Test**: Try 4, 8, 16 workers and measure throughput

### 2. Batch Size Scaling

When using multiple GPUs:
- Batch size is split across GPUs
- With 4 GPUs and batch_size=128, each GPU processes 32 samples
- Consider increasing batch size for multi-GPU training

### 3. Evaluation Frequency

For long experiments with many epochs:
- `--eval_frequency 5` can reduce total time by ~20-30%
- Still get final evaluation metrics
- Trades off intermediate monitoring for speed

### 4. GPU Selection

For shared systems:
- Check GPU availability: `nvidia-smi`
- Use specific GPUs: `--gpu_ids 2,3` to avoid busy GPUs
- Can also use `CUDA_VISIBLE_DEVICES` environment variable

## Monitoring Multi-GPU Usage

### During Training

The scripts now print GPU information:
```
Using 4 GPUs: [0, 1, 2, 3]
Device: cuda
```

### System Monitoring

Use `nvidia-smi` to monitor GPU utilization:
```bash
watch -n 1 nvidia-smi  # Update every second
```

### Expected Speedup

Theoretical vs. practical speedup with N GPUs:

| GPUs | Theoretical | Typical Actual | Efficiency |
|------|-------------|----------------|------------|
| 1    | 1.0x        | 1.0x           | 100%       |
| 2    | 2.0x        | 1.7-1.9x       | 85-95%     |
| 4    | 4.0x        | 3.0-3.6x       | 75-90%     |
| 8    | 8.0x        | 5.0-6.5x       | 62-81%     |

*Efficiency decreases due to communication overhead and data transfer*

## Troubleshooting

### Out of Memory (OOM) Errors

If you get CUDA OOM errors with multi-GPU:
1. Reduce batch size: The effective batch size is split across GPUs
2. Reduce num_workers: Each worker uses additional memory
3. Use fewer GPUs: `--gpu_ids 0,1` instead of all GPUs

### Slow Data Loading

If GPUs are underutilized (low usage in `nvidia-smi`):
1. Increase num_workers: Try 8 or 16
2. Enable pin_memory: Already enabled by default
3. Check disk I/O: Slow storage can bottleneck data loading

### No Speedup with Multiple GPUs

Potential issues:
1. Batch size too small: Increase batch size to saturate GPUs
2. Model too small: SSVAE is relatively small, may not scale perfectly
3. DataParallel overhead: For very small models, overhead dominates

## Technical Details

### DataParallel Implementation

The code uses PyTorch's `nn.DataParallel` which:
- Replicates the model on each GPU
- Splits input batch across GPUs
- Gathers outputs and computes loss
- Averages gradients across GPUs

### Memory Distribution

With 4 GPUs and batch_size=128:
- Each GPU gets 32 samples
- Model is replicated on each GPU
- Gradients are synchronized after backward pass

### Base Model Access

When using DataParallel, access the underlying model:
```python
base_model = get_base_model(wrapped_model)
```

This is handled automatically in the optimizer setup.

## Future Improvements

Potential future enhancements:
1. **DistributedDataParallel (DDP)**: More efficient than DataParallel
2. **Mixed Precision Training**: Use FP16 for faster computation
3. **Gradient Accumulation**: Simulate larger batch sizes
4. **Model Checkpointing**: Save/resume from checkpoints
5. **Early Stopping**: Stop training when metrics plateau

## Questions or Issues?

If you encounter problems with multi-GPU training:
1. Check GPU availability: `nvidia-smi`
2. Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Try single GPU first: `--gpu_ids 0`
4. Disable multi-GPU: `--no_multi_gpu`

For best results on SLURM:
- Request appropriate resources (CPUs, GPUs, memory)
- Monitor job with `squeue -u $USER`
- Check logs in the `logs/` directory
