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
import random as python_random
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from tqdm import tqdm

from ssvae_model import (
    Encoder,
    Decoder,
    elbo,
    move_tensors_to_device,
    get_device,
    setup_multi_gpu,
    get_base_model,
)
from utils import (
    get_data_loaders,
    train_epoch,
    test_epoch,
    corrupt_labels,
    UniversalEncoder,
)
from metrics import evaluate_all_metrics


def set_random_seed(seed):
    """
    Set random seed for reproducibility across all libraries

    Args:
        seed: Random seed value
    """
    python_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Default hyperparameters
DEFAULT_CONFIG = {
    # Model parameters
    "num_pixels": 784,
    "num_hidden": 256,
    "num_digits": 10,
    "num_style": 10,
    # Training parameters
    "num_samples": 8,
    "num_batch": 256,
    "num_epochs": 40,  # Moderate training for parameter sweep
    "learning_rate": 1e-3,
    "beta1": 0.90,
    "eps": 1e-9,
    "device": "auto",  # 'auto', 'cuda', 'mps', or 'cpu'
    # Multi-GPU settings
    "use_multi_gpu": False,  # Enable multi-GPU if available
    "gpu_ids": None,  # None = use all GPUs, or list like [0, 1, 2, 3]
    # Data loading settings
    "num_workers": 4,  # Number of data loading workers (increase for faster loading)
    "pin_memory": True,  # Pin memory for faster GPU transfer
    # Performance settings
    "eval_frequency": 5,  # Evaluate every N epochs (set >1 to speed up training)
    # Experiment parameters - 3D sweep
    "n_labels": [100, 600, 1000, 3000],  #
    "corruption_rates": [0.0],
    "alpha_values": [0.1, 0.5, 1, 50, 100],
    "datasets": ["MNIST"],  # , "FashionMNIST"
    "num_seeds": 10,  # Number of random seeds per configuration
    "random_seeds": list(range(42, 52)),  # Seeds: 42, 43, 44, ..., 51
    # Path parameters
    "data_path": "../data",
    "results_path": "../results/combined",
    "weights_path": "../weights/combined",
}


def run_single_combined_experiment(
    dataset_name, n_labels, corruption_rate, alpha, seed, config
):
    """
    Run a single experiment with specific label fraction, corruption rate, and alpha

    Args:
        dataset_name: 'MNIST' or 'FashionMNIST'
        n_labels: Number of labeled datapoints (e.g., 0.1 for 10%)
        corruption_rate: Fraction of labels to corrupt (e.g., 0.1 for 10%)
        alpha: Alpha parameter for ELBO (tradeoff between supervised/unsupervised)
        seed: Random seed for reproducibility
        config: Configuration dictionary

    Returns:
        results: Dictionary containing metrics and performance
    """
    # Set random seed for reproducibility
    set_random_seed(seed)

    # Determine device
    if config["device"] == "auto":
        device = get_device()
    else:
        device = torch.device(config["device"])

    print(f"\n{'='*80}")
    print(
        f"Dataset: {dataset_name} | Labels: {n_labels}% | "
        f"Corruption: {corruption_rate*100:.1f}% | Alpha: {alpha:.2f} | Seed: {seed}"
    )
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Get data loaders with optimized settings
    train_loader, test_loader = get_data_loaders(
        dataset_name=dataset_name,
        data_path=config["data_path"],
        batch_size=config["num_batch"],
        num_workers=config.get("num_workers", 4),
        download=True,
        pin_memory=config.get("pin_memory", True),
    )

    # Initialize models
    enc = Encoder(
        num_pixels=config["num_pixels"],
        num_hidden=config["num_hidden"],
        num_digits=config["num_digits"],
        num_style=config["num_style"],
        num_batch=config["num_batch"],
    )

    dec = Decoder(
        num_pixels=config["num_pixels"],
        num_hidden=config["num_hidden"],
        num_digits=config["num_digits"],
        num_style=config["num_style"],
    )

    # Move models to device
    enc.to(device)
    dec.to(device)
    move_tensors_to_device(enc, device)
    move_tensors_to_device(dec, device)

    # Setup multi-GPU if enabled and available
    is_multi_gpu = False
    if config.get("use_multi_gpu", True) and torch.cuda.is_available():
        gpu_ids = config.get("gpu_ids", None)
        enc, is_multi_gpu_enc = setup_multi_gpu(enc, gpu_ids)
        dec, is_multi_gpu_dec = setup_multi_gpu(dec, gpu_ids)
        is_multi_gpu = is_multi_gpu_enc or is_multi_gpu_dec

    # Optimizer (use base model parameters if multi-GPU)
    enc_params = get_base_model(enc).parameters()
    dec_params = get_base_model(dec).parameters()
    optimizer = torch.optim.Adam(
        list(enc_params) + list(dec_params),
        lr=config["learning_rate"],
        betas=(config["beta1"], 0.999),
    )

    # Training with custom alpha
    label_mask = {}
    train_elbos = []
    test_elbos = []
    test_accuracies = []
    eval_frequency = config.get("eval_frequency", 1)

    pbar = tqdm(range(config["num_epochs"]), desc=f"Training (α={alpha:.2f})")
    for epoch in pbar:

        # Train with specified alpha
        train_elbo, label_mask = train_epoch(
            train_loader,
            enc,
            dec,
            optimizer,
            label_mask,
            label_fraction=n_labels / len(train_loader.dataset),  # Convert to fraction
            num_samples=config["num_samples"],
            num_batch=config["num_batch"],
            num_pixels=config["num_pixels"],
            num_digits=config["num_digits"],
            device=str(device),
            eps=config["eps"],
            corruption_rate=corruption_rate,
            alpha=alpha,  # Pass alpha to training
        )
        train_elbos.append(train_elbo)

        # Test only at specified frequency or last epoch
        should_evaluate = (epoch % eval_frequency == 0) or (
            epoch == config["num_epochs"] - 1
        )

        if should_evaluate:
            test_elbo, test_accuracy = test_epoch(
                test_loader,
                enc,
                dec,
                num_samples=config["num_samples"],
                num_batch=config["num_batch"],
                num_pixels=config["num_pixels"],
                device=str(device),
                infer=True,
                alpha=alpha,  # Pass alpha to testing
            )

            test_elbos.append(test_elbo)
            test_accuracies.append(test_accuracy)

            # Update progress bar with latest metrics
            pbar.set_postfix(
                {
                    "Train ELBO": f"{train_elbo:.3e}",
                    "Test Acc": f"{test_accuracy:.3f}",
                    "Test ELBO": f"{test_elbo:.3e}",
                }
            )
        else:
            # If not evaluating, use placeholder (will be same length as train_elbos)
            # Note: These are not actual evaluations, just placeholders for consistency
            test_elbos.append(test_elbos[-1] if test_elbos else 0.0)
            test_accuracies.append(test_accuracies[-1] if test_accuracies else 0.0)
            pbar.set_postfix(
                {
                    "Train ELBO": f"{train_elbo:.3e}",
                    "Test Acc": f"(skipped)",
                }
            )

    # Final evaluation
    print("\nFinal evaluation...")
    final_test_elbo, final_test_accuracy = test_epoch(
        test_loader,
        enc,
        dec,
        num_samples=config["num_samples"],
        num_batch=config["num_batch"],
        num_pixels=config["num_pixels"],
        device=str(device),
        infer=True,
        alpha=alpha,
    )

    # Compute disentanglement metrics
    print("Computing disentanglement metrics...")
    disentanglement_metrics = evaluate_all_metrics(
        enc,
        test_loader,
        num_samples=5000,
        num_batch=config["num_batch"],
        num_pixels=config["num_pixels"],
        device=str(device),
        num_style=config["num_style"],
    )

    # Store results
    results = {
        "dataset": dataset_name,
        "n_labels": n_labels,
        "corruption_rate": corruption_rate,
        "alpha": alpha,
        "seed": seed,
        "final_test_elbo": final_test_elbo,
        "final_test_accuracy": final_test_accuracy,
        "train_elbos": train_elbos,
        "test_elbos": test_elbos,
        "test_accuracies": test_accuracies,
        "disentanglement_metrics": {
            "beta_vae": disentanglement_metrics["beta_vae"],
            "factor_vae": disentanglement_metrics["factor_vae"],
            "mig": disentanglement_metrics["mig"],
        },
        "config": config,
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
    Each configuration is run with multiple random seeds for statistical robustness

    Args:
        dataset_name: 'MNIST' or 'FashionMNIST'
        config: Configuration dictionary

    Returns:
        results_list: List of results for all parameter combinations and seeds
    """
    print(f"\n{'#'*80}")
    print(f"COMBINED 3D PARAMETER SWEEP - {dataset_name}")
    print(f"{'#'*80}")
    print(f"\nParameter Space:")
    print(f"  Number of Labels: {config['n_labels']}")
    print(f"  Corruption Rates: {config['corruption_rates']}")
    print(f"  Alpha Values: {config['alpha_values']}")
    print(
        f"  Random Seeds: {config['num_seeds']} seeds ({config['random_seeds'][0]} to {config['random_seeds'][-1]})"
    )

    total_experiments = (
        len(config["n_labels"])
        * len(config["corruption_rates"])
        * len(config["alpha_values"])
        * config["num_seeds"]
    )
    print(f"  Total Experiments: {total_experiments}\n")

    results_list = []
    failed_configs = []

    # Create checkpoints directory
    checkpoints_dir = os.path.join(config["results_path"], "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Create error log file
    error_log_file = os.path.join(config["results_path"], f"{dataset_name}_errors.log")

    # Generate all combinations by configuration (not including seeds yet)
    config_combinations = list(
        product(
            config["n_labels"],
            config["corruption_rates"],
            config["alpha_values"],
        )
    )

    total_configs = len(config_combinations)
    print(
        f"\nRunning {total_configs} configurations, each with {config['num_seeds']} seeds...\n"
    )

    config_pbar = tqdm(
        config_combinations, desc=f"{dataset_name} Configurations", position=0
    )
    for config_idx, (n_labels, corrupt_rate, alpha) in enumerate(config_pbar, 1):
        config_pbar.set_description(
            f"{dataset_name} Config {config_idx}/{total_configs} | Labels:{n_labels}% Noise:{corrupt_rate*100:.0f}% α:{alpha:.2f}"
        )

        try:
            # Check if this configuration already has a checkpoint
            checkpoint_file = os.path.join(
                checkpoints_dir,
                f"{dataset_name}_lf{n_labels}_cr{corrupt_rate:.2f}_a{alpha:.2f}.json",
            )

            if os.path.exists(checkpoint_file):
                print(
                    f"\nLoading checkpoint for config: Labels={n_labels}%, Corruption={corrupt_rate*100:.1f}%, Alpha={alpha:.2f}"
                )
                with open(checkpoint_file, "r") as f:
                    config_results = json.load(f)
                results_list.extend(config_results)
                continue

            # Run all seeds for this configuration
            config_results = []
            seed_pbar = tqdm(
                config["random_seeds"], desc=f"  Seeds", position=1, leave=False
            )
            for seed in seed_pbar:
                seed_pbar.set_description(f"  Seed {seed}")

                results = run_single_combined_experiment(
                    dataset_name, n_labels, corrupt_rate, alpha, seed, config
                )
                config_results.append(results)

            # Save checkpoint for this configuration
            serializable_config_results = []
            for r in config_results:
                r_copy = r.copy()
                r_copy["train_elbos"] = [float(x) for x in r["train_elbos"]]
                r_copy["test_elbos"] = [float(x) for x in r["test_elbos"]]
                r_copy["test_accuracies"] = [float(x) for x in r["test_accuracies"]]
                serializable_config_results.append(r_copy)

            with open(checkpoint_file, "w") as f:
                json.dump(
                    serializable_config_results, f, indent=2, cls=UniversalEncoder
                )

            print(f"\n✓ Checkpoint saved: {checkpoint_file}")

            # Add to overall results
            results_list.extend(config_results)

        except Exception as e:
            # Log the error
            error_msg = f"\n{'='*80}\n"
            error_msg += f"ERROR in configuration:\n"
            error_msg += f"  Dataset: {dataset_name}\n"
            error_msg += f"  Label Fraction: {n_labels}%\n"
            error_msg += f"  Corruption Rate: {corrupt_rate*100:.1f}%\n"
            error_msg += f"  Alpha: {alpha:.2f}\n"
            error_msg += f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            error_msg += f"  Error: {str(e)}\n"
            error_msg += f"  Error Type: {type(e).__name__}\n"
            error_msg += f"{'='*80}\n"

            # Write to error log
            with open(error_log_file, "a") as f:
                f.write(error_msg)

            # Print error message
            print(f"\nERROR: Configuration failed!")
            print(
                f"   Labels={n_labels}%, Corruption={corrupt_rate*100:.1f}%, Alpha={alpha:.2f}"
            )
            print(f"   Error: {str(e)}")
            print(f"   See {error_log_file} for details")
            print(f"   Continuing with next configuration...\n")

            # Track failed config
            failed_configs.append(
                {
                    "n_labels": n_labels,
                    "corruption_rate": corrupt_rate,
                    "alpha": alpha,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

    # Print summary of failed configurations
    if failed_configs:
        print(f"\n{'!'*80}")
        print(f"WARNING: {len(failed_configs)} configuration(s) failed:")
        for fc in failed_configs:
            print(
                f"  - Labels={fc['n_labels']}%, "
                f"Corruption={fc['corruption_rate']*100:.1f}%, "
                f"Alpha={fc['alpha']:.2f} ({fc['error_type']})"
            )
        print(f"Check {error_log_file} for details")
        print(f"{'!'*80}\n")

    return results_list, failed_configs


def plot_combined_results(results, dataset_name, save_path):
    """
    Create comprehensive visualizations for 3D parameter sweep
    Aggregates results across random seeds (mean ± std)

    Args:
        results: List of results from all experiments
        dataset_name: Name of the dataset
        save_path: Path to save plots
    """
    # Group results by configuration (excluding seed) and compute statistics
    from collections import defaultdict

    config_results = defaultdict(list)
    for r in results:
        key = (r["n_labels"], r["corruption_rate"], r["alpha"])
        config_results[key].append(r)

    # Compute mean and std for each configuration
    data = {
        "n_labels": [],
        "corruption_rate": [],
        "alpha": [],
        "accuracy_mean": [],
        "accuracy_std": [],
        "beta_vae_mean": [],
        "beta_vae_std": [],
        "mig_mean": [],
        "mig_std": [],
    }

    for (lf, cr, alpha), runs in config_results.items():
        data["n_labels"].append(lf)
        data["corruption_rate"].append(cr)
        data["alpha"].append(alpha)

        accs = [r["final_test_accuracy"] for r in runs]
        betas = [r["disentanglement_metrics"]["beta_vae"] for r in runs]
        migs = [r["disentanglement_metrics"]["mig"] for r in runs]

        data["accuracy_mean"].append(np.mean(accs))
        data["accuracy_std"].append(np.std(accs))
        data["beta_vae_mean"].append(np.mean(betas))
        data["beta_vae_std"].append(np.std(betas))
        data["mig_mean"].append(np.mean(migs))
        data["mig_std"].append(np.std(migs))

    # Create heatmaps for each alpha value
    unique_alphas = sorted(set(data["alpha"]))

    fig, axes = plt.subplots(
        len(unique_alphas), 3, figsize=(18, 6 * len(unique_alphas))
    )
    if len(unique_alphas) == 1:
        axes = axes.reshape(1, -1)

    for i, alpha_val in enumerate(unique_alphas):
        # Filter data for this alpha
        mask = np.array(data["alpha"]) == alpha_val

        # Get unique values
        n_labels = sorted(set(np.array(data["n_labels"])[mask]))
        corrupt_rates = sorted(set(np.array(data["corruption_rate"])[mask]))

        # Create matrices for mean values
        acc_matrix = np.zeros((len(corrupt_rates), len(n_labels)))
        beta_matrix = np.zeros((len(corrupt_rates), len(n_labels)))
        mig_matrix = np.zeros((len(corrupt_rates), len(n_labels)))

        # Fill matrices with aggregated mean values
        for idx in range(len(data["alpha"])):
            if data["alpha"][idx] == alpha_val:
                lf_idx = n_labels.index(data["n_labels"][idx])
                cr_idx = corrupt_rates.index(data["corruption_rate"][idx])
                acc_matrix[cr_idx, lf_idx] = data["accuracy_mean"][idx]
                beta_matrix[cr_idx, lf_idx] = data["beta_vae_mean"][idx]
                mig_matrix[cr_idx, lf_idx] = data["mig_mean"][idx]

        # Plot heatmaps
        # Accuracy
        sns.heatmap(
            acc_matrix,
            ax=axes[i, 0],
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            xticklabels=[f"{lf}%" for lf in n_labels],
            yticklabels=[f"{cr*100:.1f}%" for cr in corrupt_rates],
        )
        axes[i, 0].set_title(
            f"{dataset_name}: Accuracy (α={alpha_val:.1f}) - Mean across seeds",
            fontsize=12,
            fontweight="bold",
        )
        axes[i, 0].set_xlabel("Label Fraction")
        axes[i, 0].set_ylabel("Corruption Rate")

        # Beta-VAE
        sns.heatmap(
            beta_matrix,
            ax=axes[i, 1],
            annot=True,
            fmt=".3f",
            cmap="viridis",
            xticklabels=[f"{lf}%" for lf in n_labels],
            yticklabels=[f"{cr*100:.1f}%" for cr in corrupt_rates],
        )
        axes[i, 1].set_title(
            f"{dataset_name}: Beta-VAE Score (α={alpha_val:.1f}) - Mean across seeds",
            fontsize=12,
            fontweight="bold",
        )
        axes[i, 1].set_xlabel("Label Fraction")
        axes[i, 1].set_ylabel("Corruption Rate")

        # MIG
        sns.heatmap(
            mig_matrix,
            ax=axes[i, 2],
            annot=True,
            fmt=".3f",
            cmap="plasma",
            xticklabels=[f"{lf}%" for lf in n_labels],
            yticklabels=[f"{cr*100:.1f}%" for cr in corrupt_rates],
        )
        axes[i, 2].set_title(
            f"{dataset_name}: MIG Score (α={alpha_val:.1f}) - Mean across seeds",
            fontsize=12,
            fontweight="bold",
        )
        axes[i, 2].set_xlabel("Label Fraction")
        axes[i, 2].set_ylabel("Corruption Rate")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"{dataset_name}_combined_heatmaps.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"Saved heatmaps to {os.path.join(save_path, f'{dataset_name}_combined_heatmaps.png')}"
    )
    plt.close()

    # Create 3D interaction plots
    fig = plt.figure(figsize=(20, 6))

    # Plot 1: Alpha vs Accuracy for different corruption rates (at max label fraction)
    ax1 = fig.add_subplot(131)
    max_n_labels = max(data["n_labels"])
    for cr in sorted(set(data["corruption_rate"])):
        mask = (np.array(data["corruption_rate"]) == cr) & (
            np.array(data["n_labels"]) == max_n_labels
        )
        alphas = np.array(data["alpha"])[mask]
        accs_mean = np.array(data["accuracy_mean"])[mask]
        accs_std = np.array(data["accuracy_std"])[mask]
        ax1.errorbar(
            alphas,
            accs_mean,
            yerr=accs_std,
            fmt="o-",
            label=f"Corrupt: {cr*100:.0f}%",
            linewidth=2,
            markersize=8,
            capsize=5,
        )
    ax1.set_xlabel("Alpha", fontsize=12)
    ax1.set_ylabel("Test Accuracy", fontsize=12)
    ax1.set_title(
        f"{dataset_name}: Alpha Effect on Accuracy\n(Number of Labels: {max_n_labels})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Label Fraction vs Beta-VAE for different alphas (at zero corruption)
    ax2 = fig.add_subplot(132)
    for alpha_val in sorted(set(data["alpha"])):
        mask = (np.array(data["alpha"]) == alpha_val) & (
            np.array(data["corruption_rate"]) == 0.0
        )
        n_labels = np.array(data["n_labels"])[mask]
        betas_mean = np.array(data["beta_vae_mean"])[mask]
        betas_std = np.array(data["beta_vae_std"])[mask]
        sort_idx = np.argsort(n_labels)
        ax2.errorbar(
            np.array(n_labels)[sort_idx] * 100,
            np.array(betas_mean)[sort_idx],
            yerr=np.array(betas_std)[sort_idx],
            fmt="o-",
            label=f"α={alpha_val:.1f}",
            linewidth=2,
            markersize=8,
            capsize=5,
        )
    ax2.set_xlabel("Number of Labels", fontsize=12)
    ax2.set_ylabel("Beta-VAE Score", fontsize=12)
    ax2.set_title(
        f"{dataset_name}: Number of Labels Effect on Disentanglement\n(No Corruption)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    # Plot 3: Corruption vs MIG for different alphas (at max number of labels)
    ax3 = fig.add_subplot(133)
    for alpha_val in sorted(set(data["alpha"])):
        mask = (np.array(data["alpha"]) == alpha_val) & (
            np.array(data["n_labels"]) == max_n_labels
        )
        crs = np.array(data["corruption_rate"])[mask]
        migs_mean = np.array(data["mig_mean"])[mask]
        migs_std = np.array(data["mig_std"])[mask]
        sort_idx = np.argsort(crs)
        ax3.errorbar(
            np.array(crs)[sort_idx] * 100,
            np.array(migs_mean)[sort_idx],
            yerr=np.array(migs_std)[sort_idx],
            fmt="o-",
            label=f"α={alpha_val:.1f}",
            linewidth=2,
            markersize=8,
            capsize=5,
        )
    ax3.set_xlabel("Corruption Rate (%)", fontsize=12)
    ax3.set_ylabel("MIG Score", fontsize=12)
    ax3.set_title(
        f"{dataset_name}: Corruption Effect on MIG\n(Number of Labels: {max_n_labels})",
        fontsize=12,
        fontweight="bold",
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"{args.name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved interaction plots to {os.path.join(save_path, f'{args.name}.png')}")
    plt.close()


def main(args):
    """Main function to run all combined experiments"""

    # Create configuration
    config = DEFAULT_CONFIG.copy()
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs
    if args.num_style:
        config["num_style"] = args.num_style
    if args.device:
        config["device"] = args.device

    # Override parameter ranges if specified
    if args.n_labels:
        config["n_labels"] = [int(x) for x in args.n_labels.split(",")]
    if args.corruption_rates:
        config["corruption_rates"] = [
            float(x) for x in args.corruption_rates.split(",")
        ]
    if args.alpha_values:
        config["alpha_values"] = [float(x) for x in args.alpha_values.split(",")]
    if args.num_seeds:
        config["num_seeds"] = args.num_seeds
        config["random_seeds"] = list(range(42, 42 + args.num_seeds))

    # Performance and multi-GPU settings
    config["num_workers"] = args.num_workers
    config["use_multi_gpu"] = not args.no_multi_gpu
    if args.gpu_ids:
        config["gpu_ids"] = [int(x) for x in args.gpu_ids.split(",")]
    config["eval_frequency"] = args.eval_frequency

    # Create directories
    os.makedirs(config["results_path"], exist_ok=True)
    os.makedirs(config["weights_path"], exist_ok=True)

    # Select datasets to run
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = config["datasets"]

    all_results = {}
    all_failed_configs = {}

    for dataset_name in datasets:
        print(f"\n\n{'='*80}")
        print(f"STARTING COMBINED SWEEP FOR {dataset_name}")
        print(f"{'='*80}\n")

        # Run combined sweep
        results, failed_configs = run_combined_sweep(dataset_name, config)
        all_results[dataset_name] = results
        all_failed_configs[dataset_name] = failed_configs

        # Save results
        results_file = os.path.join(config["results_path"], f"{args.name}.json")
        with open(results_file, "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = []
            for r in results:
                r_copy = r.copy()
                r_copy["train_elbos"] = [float(x) for x in r["train_elbos"]]
                r_copy["test_elbos"] = [float(x) for x in r["test_elbos"]]
                r_copy["test_accuracies"] = [float(x) for x in r["test_accuracies"]]
                serializable_results.append(r_copy)
            json.dump(serializable_results, f, indent=2, cls=UniversalEncoder)
        print(f"Saved results to {results_file}")

        # Plot results
        plot_combined_results(results, dataset_name, config["results_path"])

    print("\n\n" + "=" * 80)
    print("ALL COMBINED EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved to: {config['results_path']}")
    print(
        f"Checkpoints saved to: {os.path.join(config['results_path'], 'checkpoints')}"
    )

    # Print summary of all failures
    total_failures = sum(len(fails) for fails in all_failed_configs.values())
    if total_failures > 0:
        print(f"\n{'!'*80}")
        print(
            f"SUMMARY: {total_failures} total configuration(s) failed across all datasets"
        )
        for dataset, failed in all_failed_configs.items():
            if failed:
                print(f"\n{dataset}: {len(failed)} failed configuration(s)")
                print(
                    f"  See {os.path.join(config['results_path'], f'{dataset}_errors.log')} for details"
                )
        print(f"{'!'*80}")
    else:
        print("\n✓ All configurations completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined 3D Parameter Sweep for SSVAE Label Robustness"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["MNIST", "FashionMNIST"],
        help="Dataset to use (default: both)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help=f'Number of training epochs (default: {DEFAULT_CONFIG["num_epochs"]})',
    )
    parser.add_argument(
        "--num_style",
        type=int,
        help=f'Number of style latent dimensions (default: {DEFAULT_CONFIG["num_style"]})',
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--n_labels",
        type=str,
        help='Comma-separated number of labels (e.g., "100,500,1000")',
    )
    parser.add_argument(
        "--corruption_rates",
        type=str,
        help='Comma-separated corruption rates (e.g., "0.0,0.1,0.2")',
    )
    parser.add_argument(
        "--alpha_values",
        type=str,
        help='Comma-separated alpha values (e.g., "0.1,0.5,1.0")',
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        help=f'Number of random seeds per configuration (default: {DEFAULT_CONFIG["num_seeds"]})',
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--no_multi_gpu",
        action="store_true",
        help="Disable multi-GPU training even if multiple GPUs are available",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        help='Comma-separated GPU IDs to use (e.g., "0,1,2,3")',
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=1,
        help="Evaluate every N epochs (default: 1). Set higher to speed up training.",
    )
    parser.add_argument("--name", type=str, help=f"Experiment name", required=True)

    args = parser.parse_args()

    main(args)
