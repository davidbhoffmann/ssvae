#!/bin/bash
#SBATCH --job-name=ssvae_combined
#SBATCH --output=logs/combined_experiment_%j.out
#SBATCH --error=logs/combined_experiment_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Print job information
echo "========================================="
echo "SSVAE Combined Robustness Experiment"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load conda and activate environment
eval "$(conda shell.bash hook)"
conda activate vae

# Verify GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Navigate to experiment directory
cd /scratch/shared/beegfs/dhoffmann/projects/ssvae/experiments

# Run the experiment
echo "Starting combined robustness experiment..."
echo ""

python combined_robustness_experiment.py \
    --num_epochs 75 \
    --num_batch 128 \
    --num_samples 8 \
    --device cuda

echo ""
echo "Experiment completed at: $(date)"
echo "Results saved in: results/combined_robustness/"
