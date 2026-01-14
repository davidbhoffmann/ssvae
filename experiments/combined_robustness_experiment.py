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

from ssvae_model import Encoder, Decoder, elbo, move_tensors_to_device, get_device
from utils import get_data_loaders, train_epoch, test_epoch, corrupt_labels
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
    "num_style": 50,
    # Training parameters
    "num_samples": 8,
    "num_batch": 128,
    "num_epochs": 75,  # Moderate training for parameter sweep
    "learning_rate": 1e-3,
    "beta1": 0.90,
    "eps": 1e-9,
    "device": "auto",  # 'auto', 'cuda', 'mps', or 'cpu'
    # Experiment parameters - 3D sweep
    "label_fractions": [0.1, 0.05, 0.01],  # 3 levels
    "corruption_rates": [0.0, 0.1, 0.2],  # 3 levels
    "alpha_values": [0.1, 0.5, 1.0],  # 3 levels (9 combinations total per dataset)
    "datasets": ["MNIST", "FashionMNIST"],
    "num_seeds": 10,  # Number of random seeds per configuration
    "random_seeds": list(range(42, 52)),  # Seeds: 42, 43, 44, ..., 51
    # Path parameters
    "data_path": "../data",
    "results_path": "../results/combined",
    "weights_path": "../weights/combined",
}


def run_single_combined_experiment(
    dataset_name, label_fraction, corruption_rate, alpha, seed, config
):
    """
    Run a single experiment with specific label fraction, corruption rate, and alpha

    Args:
        dataset_name: 'MNIST' or 'FashionMNIST'
        label_fraction: Fraction of labeled data (e.g., 0.1 for 10%)
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
        f"Dataset: {dataset_name} | Labels: {label_fraction*100:.1f}% | "
        f"Corruption: {corruption_rate*100:.1f}% | Alpha: {alpha:.2f} | Seed: {seed}"
    )
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        dataset_name=dataset_name,
        data_path=config["data_path"],
        batch_size=config["num_batch"],
        download=True,
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

    # Optimizer
    optimizer = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=config["learning_rate"],
        betas=(config["beta1"], 0.999),
    )

    # Training with custom alpha
    label_mask = {}
    train_elbos = []
    test_elbos = []
    test_accuracies = []

    pbar = tqdm(range(config["num_epochs"]), desc=f"Training (α={alpha:.2f})")
    for epoch in pbar:
        train_start = time.time()

        # Train with specified alpha
        train_elbo, label_mask = train_epoch(
            train_loader,
            enc,
            dec,
            optimizer,
            label_mask,
            label_fraction=label_fraction,
            num_samples=config["num_samples"],
            num_batch=config["num_batch"],
            num_pixels=config["num_pixels"],
            num_digits=config["num_digits"],
            device=str(device),
            eps=config["eps"],
            corruption_rate=corruption_rate,
            alpha=alpha,  # Pass alpha to training
        )

        train_end = time.time()
        train_elbos.append(train_elbo)

        # Test
        test_start = time.time()
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
        test_end = time.time()

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
        "label_fraction": label_fraction,
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
    print(f"  Label Fractions: {config['label_fractions']}")
    print(f"  Corruption Rates: {config['corruption_rates']}")
    print(f"  Alpha Values: {config['alpha_values']}")
    print(
        f"  Random Seeds: {config['num_seeds']} seeds ({config['random_seeds'][0]} to {config['random_seeds'][-1]})"
    )

    total_experiments = (
        len(config["label_fractions"])
        * len(config["corruption_rates"])
        * len(config["alpha_values"])
        * config["num_seeds"]
    )
    print(f"  Total Experiments: {total_experiments}\n")

    results_list = []

    # Create checkpoints directory
    checkpoints_dir = os.path.join(config["results_path"], "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Generate all combinations by configuration (not including seeds yet)
    config_combinations = list(
        product(
            config["label_fractions"],
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
    for config_idx, (label_frac, corrupt_rate, alpha) in enumerate(config_pbar, 1):
        config_pbar.set_description(
            f"{dataset_name} Config {config_idx}/{total_configs} | Labels:{label_frac*100:.1f}% Noise:{corrupt_rate*100:.0f}% α:{alpha:.2f}"
        )

        # Check if this configuration already has a checkpoint
        checkpoint_file = os.path.join(
            checkpoints_dir,
            f"{dataset_name}_lf{label_frac:.3f}_cr{corrupt_rate:.2f}_a{alpha:.2f}.json",
        )

        if os.path.exists(checkpoint_file):
            print(
                f"\nLoading checkpoint for config: Labels={label_frac*100:.1f}%, Corruption={corrupt_rate*100:.1f}%, Alpha={alpha:.2f}"
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
                dataset_name, label_frac, corrupt_rate, alpha, seed, config
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
            json.dump(serializable_config_results, f, indent=2)

        print(f"\n✓ Checkpoint saved: {checkpoint_file}")

        # Add to overall results
        results_list.extend(config_results)

    return results_list


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
        key = (r["label_fraction"], r["corruption_rate"], r["alpha"])
        config_results[key].append(r)

    # Compute mean and std for each configuration
    data = {
        "label_fraction": [],
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
        data["label_fraction"].append(lf)
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
        label_fracs = sorted(set(np.array(data["label_fraction"])[mask]))
        corrupt_rates = sorted(set(np.array(data["corruption_rate"])[mask]))

        # Create matrices for mean values
        acc_matrix = np.zeros((len(corrupt_rates), len(label_fracs)))
        beta_matrix = np.zeros((len(corrupt_rates), len(label_fracs)))
        mig_matrix = np.zeros((len(corrupt_rates), len(label_fracs)))

        # Fill matrices with aggregated mean values
        for idx in range(len(data["alpha"])):
            if data["alpha"][idx] == alpha_val:
                lf_idx = label_fracs.index(data["label_fraction"][idx])
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
            xticklabels=[f"{lf*100:.1f}%" for lf in label_fracs],
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
            xticklabels=[f"{lf*100:.1f}%" for lf in label_fracs],
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
            xticklabels=[f"{lf*100:.1f}%" for lf in label_fracs],
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
    max_label_frac = max(data["label_fraction"])
    for cr in sorted(set(data["corruption_rate"])):
        mask = (np.array(data["corruption_rate"]) == cr) & (
            np.array(data["label_fraction"]) == max_label_frac
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
        f"{dataset_name}: Alpha Effect on Accuracy\n(Label Fraction: {max_label_frac*100:.0f}%)",
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
        lfs = np.array(data["label_fraction"])[mask]
        betas_mean = np.array(data["beta_vae_mean"])[mask]
        betas_std = np.array(data["beta_vae_std"])[mask]
        sort_idx = np.argsort(lfs)
        ax2.errorbar(
            np.array(lfs)[sort_idx] * 100,
            np.array(betas_mean)[sort_idx],
            yerr=np.array(betas_std)[sort_idx],
            fmt="o-",
            label=f"α={alpha_val:.1f}",
            linewidth=2,
            markersize=8,
            capsize=5,
        )
    ax2.set_xlabel("Label Fraction (%)", fontsize=12)
    ax2.set_ylabel("Beta-VAE Score", fontsize=12)
    ax2.set_title(
        f"{dataset_name}: Label Fraction Effect on Disentanglement\n(No Corruption)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    # Plot 3: Corruption vs MIG for different alphas (at max label fraction)
    ax3 = fig.add_subplot(133)
    for alpha_val in sorted(set(data["alpha"])):
        mask = (np.array(data["alpha"]) == alpha_val) & (
            np.array(data["label_fraction"]) == max_label_frac
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
        f"{dataset_name}: Corruption Effect on MIG\n(Label Fraction: {max_label_frac*100:.0f}%)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"{dataset_name}_combined_interactions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"Saved interaction plots to {os.path.join(save_path, f'{dataset_name}_combined_interactions.png')}"
    )
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
    if args.label_fractions:
        config["label_fractions"] = [float(x) for x in args.label_fractions.split(",")]
    if args.corruption_rates:
        config["corruption_rates"] = [
            float(x) for x in args.corruption_rates.split(",")
        ]
    if args.alpha_values:
        config["alpha_values"] = [float(x) for x in args.alpha_values.split(",")]
    if args.num_seeds:
        config["num_seeds"] = args.num_seeds
        config["random_seeds"] = list(range(42, 42 + args.num_seeds))

    # Create directories
    os.makedirs(config["results_path"], exist_ok=True)
    os.makedirs(config["weights_path"], exist_ok=True)

    # Select datasets to run
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = config["datasets"]

    all_results = {}

    for dataset_name in datasets:
        print(f"\n\n{'='*80}")
        print(f"STARTING COMBINED SWEEP FOR {dataset_name}")
        print(f"{'='*80}\n")

        # Run combined sweep
        results = run_combined_sweep(dataset_name, config)
        all_results[dataset_name] = results

        # Save results
        results_file = os.path.join(
            config["results_path"], f"{dataset_name}_combined_results.json"
        )
        with open(results_file, "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = []
            for r in results:
                r_copy = r.copy()
                r_copy["train_elbos"] = [float(x) for x in r["train_elbos"]]
                r_copy["test_elbos"] = [float(x) for x in r["test_elbos"]]
                r_copy["test_accuracies"] = [float(x) for x in r["test_accuracies"]]
                serializable_results.append(r_copy)
            json.dump(serializable_results, f, indent=2)
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
        "--label_fractions",
        type=str,
        help='Comma-separated label fractions (e.g., "0.1,0.05,0.01")',
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

    args = parser.parse_args()

    main(args)
