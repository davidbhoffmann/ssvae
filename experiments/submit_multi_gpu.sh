#!/bin/bash
#SBATCH --job-name=ssvae_multi_gpu
#SBATCH --output=logs/ssvae_multi_gpu_%j.out
#SBATCH --error=logs/ssvae_multi_gpu_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Print job information
echo "========================================="
echo "SSVAE Multi-GPU Experiment"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "GPUs requested: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo ""

# Initialize conda for bash shell
# NOTE: Update this path to match your conda installation
eval "$(/scratch/shared/beegfs/dhoffmann/miniconda3/condabin/conda shell.bash hook)"
conda activate vae

# Verify conda environment
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Verify GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to experiment directory
# NOTE: Update this path to match your project location
cd /scratch/shared/beegfs/dhoffmann/projects/ssvae/experiments

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the experiment with multi-GPU support
echo "Starting experiment with multi-GPU support..."
echo ""

# Example: Run combined robustness experiment with custom parameters
python combined_robustness_experiment.py \
    --dataset MNIST \
    --num_epochs 75 \
    --device auto

echo ""
echo "Experiment completed at: $(date)"
