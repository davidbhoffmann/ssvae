"""
Combined Robustness Experiment for SSVAE

This experiment explores the 3D parameter space by varying:
1. Label quantity (label fraction)
2. Label quality (corruption rate)
3. Alpha parameter (controls tradeoff between supervised and unsupervised loss)

The goal is to understand the interactions between these factors and their
combined effect on both classification performance and disentanglement quality.
"""

# Fix for Python 3.10+ compatibility with probtorch
import sys
if sys.version_info >= (3, 10):
    import collections
    import collections.abc
    collections.MutableMapping = collections.abc.MutableMapping

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from random import random
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from ssvae_model import Encoder, Decoder, elbo, move_tensors_to_device, get_device
from utils import get_data_loaders, train_epoch, test_epoch, corrupt_labels
from metrics import evaluate_all_metrics


# Default hyperparameters
DEFAULT_CONFIG = {
    # Model parameters
    'num_pixels': 784,
    'num_hidden': 256,
    'num_digits': 10,
    'num_style': 50,
    
    # Training parameters
    'num_samples': 8,
    'num_batch': 128,
    'num_epochs': 75,  # Moderate training for parameter sweep
    'learning_rate': 1e-3,
    'beta1': 0.90,
    'eps': 1e-9,
    'device': 'auto',  # 'auto', 'cuda', 'mps', or 'cpu'
    
    # Experiment parameters - 3D sweep
    'label_fractions': [0.1, 0.05, 0.01],  # 3 levels
    'corruption_rates': [0.0, 0.1, 0.2],   # 3 levels
    'alpha_values': [0.1, 0.5, 1.0],       # 3 levels (9 combinations total per dataset)
    'datasets': ['MNIST', 'FashionMNIST'],
    
    # Path parameters
    'data_path': '../data',
    'results_path': '../results/combined',
    'weights_path': '../weights/combined',
}


def run_single_combined_experiment(dataset_name, label_fraction, corruption_rate, 
                                   alpha, config):
    """
    Run a single experiment with specific label fraction, corruption rate, and alpha
    
    Args:
        dataset_name: 'MNIST' or 'FashionMNIST'
        label_fraction: Fraction of labeled data (e.g., 0.1 for 10%)
        corruption_rate: Fraction of labels to corrupt (e.g., 0.1 for 10%)
        alpha: Alpha parameter for ELBO (tradeoff between supervised/unsupervised)
        config: Configuration dictionary
    
    Returns:
        results: Dictionary containing metrics and performance
    """
    # Determine device
    if config['device'] == 'auto':
        device = get_device()
    else:
        device = torch.device(config['device'])
    
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name} | Labels: {label_fraction*100:.1f}% | "
          f"Corruption: {corruption_rate*100:.1f}% | Alpha: {alpha:.2f}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        dataset_name=dataset_name,
        data_path=config['data_path'],
        batch_size=config['num_batch'],
        download=True
    )
    
    # Initialize models
    enc = Encoder(
        num_pixels=config['num_pixels'],
        num_hidden=config['num_hidden'],
        num_digits=config['num_digits'],
        num_style=config['num_style'],
        num_batch=config['num_batch']
    )
    
    dec = Decoder(
        num_pixels=config['num_pixels'],
        num_hidden=config['num_hidden'],
        num_digits=config['num_digits'],
        num_style=config['num_style']
    )
    
    # Move models to device
    enc.to(device)
    dec.to(device)
    move_tensors_to_device(enc, device)
    move_tensors_to_device(dec, device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=config['learning_rate'],
        betas=(config['beta1'], 0.999)
    )
    
    # Training with custom alpha
    label_mask = {}
    train_elbos = []
    test_elbos = []
    test_accuracies = []
    
    for epoch in range(config['num_epochs']):
        train_start = time.time()
        
        # Train with specified alpha
        train_elbo, label_mask = train_epoch(
            train_loader, enc, dec, optimizer, label_mask,
            label_fraction=label_fraction,
            num_samples=config['num_samples'],
            num_batch=config['num_batch'],
            num_pixels=config['num_pixels'],
            num_digits=config['num_digits'],
            device=str(device),
            eps=config['eps'],
            corruption_rate=corruption_rate,
            alpha=alpha  # Pass alpha to training
        )
        
        train_end = time.time()
        train_elbos.append(train_elbo)
        
        # Test
        test_start = time.time()
        test_elbo, test_accuracy = test_epoch(
            test_loader, enc, dec,
            num_samples=config['num_samples'],
            num_batch=config['num_batch'],
            num_pixels=config['num_pixels'],
            device=str(device),
            infer=True,
            alpha=alpha  # Pass alpha to testing
        )
        test_end = time.time()
        
        test_elbos.append(test_elbo)
        test_accuracies.append(test_accuracy)
        
        # Print progress every 15 epochs
        if (epoch + 1) % 15 == 0:
            print(f'[Epoch {epoch+1}/{config["num_epochs"]}] '
                  f'Train ELBO: {train_elbo:.4e} ({train_end-train_start:.1f}s) | '
                  f'Test ELBO: {test_elbo:.4e}, Acc: {test_accuracy:.3f} ({test_end-test_start:.1f}s)')
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_test_elbo, final_test_accuracy = test_epoch(
        test_loader, enc, dec,
        num_samples=config['num_samples'],
        num_batch=config['num_batch'],
        num_pixels=config['num_pixels'],
        device=str(device),
        infer=True,
        alpha=alpha
    )
    
    # Compute disentanglement metrics
    print("Computing disentanglement metrics...")
    disentanglement_metrics = evaluate_all_metrics(
        enc, test_loader,
        num_samples=5000,
        num_batch=config['num_batch'],
        num_pixels=config['num_pixels'],
        device=str(device),
        num_style=config['num_style']
    )
    
    # Store results
    results = {
        'dataset': dataset_name,
        'label_fraction': label_fraction,
        'corruption_rate': corruption_rate,
        'alpha': alpha,
        'final_test_elbo': final_test_elbo,
        'final_test_accuracy': final_test_accuracy,
        'train_elbos': train_elbos,
        'test_elbos': test_elbos,
        'test_accuracies': test_accuracies,
        'disentanglement_metrics': {
            'beta_vae': disentanglement_metrics['beta_vae'],
            'factor_vae': disentanglement_metrics['factor_vae'],
            'mig': disentanglement_metrics['mig'],
        },
        'config': config
    }
    
    print(f"\nResults:")
    print(f"  Test Accuracy: {final_test_accuracy:.3f}")
    print(f"  Test ELBO: {final_test_elbo:.4e}")
    print(f"  Beta-VAE Score: {disentanglement_metrics['beta_vae']:.4f}")
    print(f"  Factor-VAE Score: {disentanglement_metrics['factor_vae']:.4f}")
    print(f"  MIG Score: {disentanglement_metrics['mig']:.4f}")
    
    return results


def run_combined_sweep(dataset_name, config):
    """
    Run full 3D parameter sweep varying label fraction, corruption, and alpha
    
    Args:
        dataset_name: 'MNIST' or 'FashionMNIST'
        config: Configuration dictionary
    
    Returns:
        results_list: List of results for all parameter combinations
    """
    print(f"\n{'#'*80}")
    print(f"COMBINED 3D PARAMETER SWEEP - {dataset_name}")
    print(f"{'#'*80}")
    print(f"\nParameter Space:")
    print(f"  Label Fractions: {config['label_fractions']}")
    print(f"  Corruption Rates: {config['corruption_rates']}")
    print(f"  Alpha Values: {config['alpha_values']}")
    
    total_experiments = (len(config['label_fractions']) * 
                        len(config['corruption_rates']) * 
                        len(config['alpha_values']))
    print(f"  Total Experiments: {total_experiments}\n")
    
    results_list = []
    experiment_num = 0
    
    # Generate all combinations
    for label_frac, corrupt_rate, alpha in product(
        config['label_fractions'],
        config['corruption_rates'],
        config['alpha_values']
    ):
        experiment_num += 1
        print(f"\n{'*'*80}")
        print(f"Experiment {experiment_num}/{total_experiments}")
        print(f"{'*'*80}")
        
        results = run_single_combined_experiment(
            dataset_name, label_frac, corrupt_rate, alpha, config
        )
        results_list.append(results)
    
    return results_list


def plot_combined_results(results, dataset_name, save_path):
    """
    Create comprehensive visualizations for 3D parameter sweep
    
    Args:
        results: List of results from all experiments
        dataset_name: Name of the dataset
        save_path: Path to save plots
    """
    # Convert results to structured format
    data = {
        'label_fraction': [],
        'corruption_rate': [],
        'alpha': [],
        'accuracy': [],
        'beta_vae': [],
        'mig': [],
    }
    
    for r in results:
        data['label_fraction'].append(r['label_fraction'])
        data['corruption_rate'].append(r['corruption_rate'])
        data['alpha'].append(r['alpha'])
        data['accuracy'].append(r['final_test_accuracy'])
        data['beta_vae'].append(r['disentanglement_metrics']['beta_vae'])
        data['mig'].append(r['disentanglement_metrics']['mig'])
    
    # Create heatmaps for each alpha value
    unique_alphas = sorted(set(data['alpha']))
    
    fig, axes = plt.subplots(len(unique_alphas), 3, figsize=(18, 6*len(unique_alphas)))
    if len(unique_alphas) == 1:
        axes = axes.reshape(1, -1)
    
    for i, alpha_val in enumerate(unique_alphas):
        # Filter data for this alpha
        mask = np.array(data['alpha']) == alpha_val
        
        # Get unique values
        label_fracs = sorted(set(np.array(data['label_fraction'])[mask]))
        corrupt_rates = sorted(set(np.array(data['corruption_rate'])[mask]))
        
        # Create matrices
        acc_matrix = np.zeros((len(corrupt_rates), len(label_fracs)))
        beta_matrix = np.zeros((len(corrupt_rates), len(label_fracs)))
        mig_matrix = np.zeros((len(corrupt_rates), len(label_fracs)))
        
        for j, r in enumerate(results):
            if r['alpha'] == alpha_val:
                lf_idx = label_fracs.index(r['label_fraction'])
                cr_idx = corrupt_rates.index(r['corruption_rate'])
                acc_matrix[cr_idx, lf_idx] = r['final_test_accuracy']
                beta_matrix[cr_idx, lf_idx] = r['disentanglement_metrics']['beta_vae']
                mig_matrix[cr_idx, lf_idx] = r['disentanglement_metrics']['mig']
        
        # Plot heatmaps
        # Accuracy
        sns.heatmap(acc_matrix, ax=axes[i, 0], annot=True, fmt='.3f', 
                   cmap='RdYlGn', vmin=0, vmax=1,
                   xticklabels=[f'{lf*100:.1f}%' for lf in label_fracs],
                   yticklabels=[f'{cr*100:.1f}%' for cr in corrupt_rates])
        axes[i, 0].set_title(f'{dataset_name}: Accuracy (α={alpha_val:.1f})', 
                            fontsize=12, fontweight='bold')
        axes[i, 0].set_xlabel('Label Fraction')
        axes[i, 0].set_ylabel('Corruption Rate')
        
        # Beta-VAE
        sns.heatmap(beta_matrix, ax=axes[i, 1], annot=True, fmt='.3f',
                   cmap='viridis',
                   xticklabels=[f'{lf*100:.1f}%' for lf in label_fracs],
                   yticklabels=[f'{cr*100:.1f}%' for cr in corrupt_rates])
        axes[i, 1].set_title(f'{dataset_name}: Beta-VAE Score (α={alpha_val:.1f})',
                            fontsize=12, fontweight='bold')
        axes[i, 1].set_xlabel('Label Fraction')
        axes[i, 1].set_ylabel('Corruption Rate')
        
        # MIG
        sns.heatmap(mig_matrix, ax=axes[i, 2], annot=True, fmt='.3f',
                   cmap='plasma',
                   xticklabels=[f'{lf*100:.1f}%' for lf in label_fracs],
                   yticklabels=[f'{cr*100:.1f}%' for cr in corrupt_rates])
        axes[i, 2].set_title(f'{dataset_name}: MIG Score (α={alpha_val:.1f})',
                            fontsize=12, fontweight='bold')
        axes[i, 2].set_xlabel('Label Fraction')
        axes[i, 2].set_ylabel('Corruption Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{dataset_name}_combined_heatmaps.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved heatmaps to {os.path.join(save_path, f'{dataset_name}_combined_heatmaps.png')}")
    plt.close()
    
    # Create 3D interaction plots
    fig = plt.figure(figsize=(20, 6))
    
    # Plot 1: Alpha vs Accuracy for different corruption rates (at max label fraction)
    ax1 = fig.add_subplot(131)
    max_label_frac = max(data['label_fraction'])
    for cr in sorted(set(data['corruption_rate'])):
        mask = (np.array(data['corruption_rate']) == cr) & \
               (np.array(data['label_fraction']) == max_label_frac)
        alphas = np.array(data['alpha'])[mask]
        accs = np.array(data['accuracy'])[mask]
        ax1.plot(alphas, accs, 'o-', label=f'Corrupt: {cr*100:.0f}%', linewidth=2, markersize=8)
    ax1.set_xlabel('Alpha', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title(f'{dataset_name}: Alpha Effect on Accuracy\n(Label Fraction: {max_label_frac*100:.0f}%)',
                 fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Label Fraction vs Beta-VAE for different alphas (at zero corruption)
    ax2 = fig.add_subplot(132)
    for alpha_val in sorted(set(data['alpha'])):
        mask = (np.array(data['alpha']) == alpha_val) & \
               (np.array(data['corruption_rate']) == 0.0)
        lfs = np.array(data['label_fraction'])[mask]
        betas = np.array(data['beta_vae'])[mask]
        sort_idx = np.argsort(lfs)
        ax2.plot(np.array(lfs)[sort_idx]*100, np.array(betas)[sort_idx], 
                'o-', label=f'α={alpha_val:.1f}', linewidth=2, markersize=8)
    ax2.set_xlabel('Label Fraction (%)', fontsize=12)
    ax2.set_ylabel('Beta-VAE Score', fontsize=12)
    ax2.set_title(f'{dataset_name}: Label Fraction Effect on Disentanglement\n(No Corruption)',
                 fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Plot 3: Corruption vs MIG for different alphas (at max label fraction)
    ax3 = fig.add_subplot(133)
    for alpha_val in sorted(set(data['alpha'])):
        mask = (np.array(data['alpha']) == alpha_val) & \
               (np.array(data['label_fraction']) == max_label_frac)
        crs = np.array(data['corruption_rate'])[mask]
        migs = np.array(data['mig'])[mask]
        sort_idx = np.argsort(crs)
        ax3.plot(np.array(crs)[sort_idx]*100, np.array(migs)[sort_idx],
                'o-', label=f'α={alpha_val:.1f}', linewidth=2, markersize=8)
    ax3.set_xlabel('Corruption Rate (%)', fontsize=12)
    ax3.set_ylabel('MIG Score', fontsize=12)
    ax3.set_title(f'{dataset_name}: Corruption Effect on MIG\n(Label Fraction: {max_label_frac*100:.0f}%)',
                 fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{dataset_name}_combined_interactions.png'),
                dpi=300, bbox_inches='tight')
    print(f"Saved interaction plots to {os.path.join(save_path, f'{dataset_name}_combined_interactions.png')}")
    plt.close()


def main(args):
    """Main function to run all combined experiments"""
    
    # Create configuration
    config = DEFAULT_CONFIG.copy()
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.num_style:
        config['num_style'] = args.num_style
    if args.device:
        config['device'] = args.device
    
    # Override parameter ranges if specified
    if args.label_fractions:
        config['label_fractions'] = [float(x) for x in args.label_fractions.split(',')]
    if args.corruption_rates:
        config['corruption_rates'] = [float(x) for x in args.corruption_rates.split(',')]
    if args.alpha_values:
        config['alpha_values'] = [float(x) for x in args.alpha_values.split(',')]
    
    # Create directories
    os.makedirs(config['results_path'], exist_ok=True)
    os.makedirs(config['weights_path'], exist_ok=True)
    
    # Select datasets to run
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = config['datasets']
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n\n{'='*80}")
        print(f"STARTING COMBINED SWEEP FOR {dataset_name}")
        print(f"{'='*80}\n")
        
        # Run combined sweep
        results = run_combined_sweep(dataset_name, config)
        all_results[dataset_name] = results
        
        # Save results
        results_file = os.path.join(config['results_path'], 
                                   f'{dataset_name}_combined_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = []
            for r in results:
                r_copy = r.copy()
                r_copy['train_elbos'] = [float(x) for x in r['train_elbos']]
                r_copy['test_elbos'] = [float(x) for x in r['test_elbos']]
                r_copy['test_accuracies'] = [float(x) for x in r['test_accuracies']]
                serializable_results.append(r_copy)
            json.dump(serializable_results, f, indent=2)
        print(f"Saved results to {results_file}")
        
        # Plot results
        plot_combined_results(results, dataset_name, config['results_path'])
    
    print("\n\n" + "="*80)
    print("ALL COMBINED EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"\nResults saved to: {config['results_path']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combined 3D Parameter Sweep for SSVAE Label Robustness'
    )
    
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'FashionMNIST'], 
                       help='Dataset to use (default: both)')
    parser.add_argument('--num_epochs', type=int, 
                       help=f'Number of training epochs (default: {DEFAULT_CONFIG["num_epochs"]})')
    parser.add_argument('--num_style', type=int,
                       help=f'Number of style latent dimensions (default: {DEFAULT_CONFIG["num_style"]})')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto)')
    parser.add_argument('--label_fractions', type=str,
                       help='Comma-separated label fractions (e.g., "0.1,0.05,0.01")')
    parser.add_argument('--corruption_rates', type=str,
                       help='Comma-separated corruption rates (e.g., "0.0,0.1,0.2")')
    parser.add_argument('--alpha_values', type=str,
                       help='Comma-separated alpha values (e.g., "0.1,0.5,1.0")')
    
    args = parser.parse_args()
    
    main(args)
