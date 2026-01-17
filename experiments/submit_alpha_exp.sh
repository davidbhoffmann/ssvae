#!/bin/bash
#SBATCH --job-name=ablation_combined
#SBATCH --output=logs/combined_experiment_%j.out
#SBATCH --error=logs/combined_experiment_%j.err
#SBATCH --time=96:00:00
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

# Initialize conda for bash shell
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

# Navigate to experiment directory
echo /usr/bin/nvidia-smi
cd /scratch/shared/beegfs/dhoffmann/projects/ssvae/experiments

# Run the experiment
echo "Starting combined robustness experiment..."
echo ""


python combined_robustness_experiment.py \
    --dataset MNIST --num_epochs 40 \
    --n_labels 100,600,1000,3000 \
    --corruption_rates 0.0, \
    --alpha_values 0.1,0.5,1,10,25,50,75,100 \
    --name alpha_0_experiment \
    --no_multi_gpu

echo ""
echo "Experiment completed at: $(date)"
