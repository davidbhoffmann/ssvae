# SSVAE Label Robustness and Sparsity Analysis

This folder contains experiments to evaluate the robustness of Semi-Supervised Variational Autoencoders (SSVAE) to label quality and supervision weight.


### Quick Start 

```bash
# Python experiment
python experiment_pipeline.py \
    --dataset MNIST --num_epochs 40 \
    --n_labels 100,600,1000,3000 \
    --corruption_rates 0.0 \
    --alpha_values 0.1,0.5,1,10,25,50,75,100 \
    --name alpha_1_experiment \
    --no_multi_gpu

# SLURM submission 
sbatch submit_alpha_exp.sh
```

