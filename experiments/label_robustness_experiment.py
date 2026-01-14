"""
Label Robustness and Sparsity Analysis for SSVAE

This experiment systematically tests the "break point" of the SSVAE framework by:
1. Varying label quantity (from 10% down to 0.1%)
2. Varying label noise (corrupting 0-30% of labels)
3. Measuring disentanglement metrics vs classification accuracy

The goal is to show how sensitive disentanglement is to supervision quality.
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
from tqdm import tqdm

from ssvae_model import Encoder, Decoder, elbo, move_tensors_to_device, get_device
from utils import get_data_loaders, train_epoch, test_epoch, corrupt_labels
from metrics import evaluate_all_metrics


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
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "beta1": 0.90,
    "eps": 1e-9,
    "cuda": torch.cuda.is_available(),
    # Experiment parameters
    "label_fractions": [0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
    "corruption_rates": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "datasets": ["MNIST", "FashionMNIST"],
    # Path parameters
    "data_path": "../data",
    "results_path": "../results",
    "weights_path": "../weights",
}


def run_single_experiment(dataset_name, label_fraction, corruption_rate, config):
    """
    Run a single experiment with specific label fraction and corruption rate

    Args:
        dataset_name: 'MNIST' or 'FashionMNIST'
        label_fraction: Fraction of labeled data (e.g., 0.1 for 10%)
        corruption_rate: Fraction of labels to corrupt (e.g., 0.1 for 10%)
        config: Configuration dictionary

    Returns:
        results: Dictionary containing metrics and performance
    """
    print(f"\n{'='*80}")
    print(
        f"Dataset: {dataset_name} | Labels: {label_fraction*100:.1f}% | Corruption: {corruption_rate*100:.1f}%"
    )
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

    if config["cuda"]:
        enc.cuda()
        dec.cuda()
        cuda_tensors(enc)
        cuda_tensors(dec)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=config["learning_rate"],
        betas=(config["beta1"], 0.999),
    )

    # Training
    label_mask = {}
    train_elbos = []
    test_elbos = []
    test_accuracies = []

    pbar = tqdm(
        range(config["num_epochs"]),
        desc=f"Labels:{label_fraction*100:.1f}% Noise:{corruption_rate*100:.0f}%",
    )
    for epoch in pbar:
        train_start = time.time()

        # Train
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
        )
        test_end = time.time()

        test_elbos.append(test_elbo)
        test_accuracies.append(test_accuracy)

        # Update progress bar
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
    )

    # Compute disentanglement metrics
    print("Computing disentanglement metrics...")
    disentanglement_metrics = evaluate_all_metrics(
        enc,
        test_loader,
        num_samples=5000,
        num_batch=config["num_batch"],
        num_pixels=config["num_pixels"],
        cuda=config["cuda"],
        num_style=config["num_style"],
    )

    # Store results
    results = {
        "dataset": dataset_name,
        "label_fraction": label_fraction,
        "corruption_rate": corruption_rate,
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


def run_label_sparsity_experiment(dataset_name, config):
    """
    Run experiment varying label fraction with no corruption

    Args:
        dataset_name: 'MNIST' or 'FashionMNIST'
        config: Configuration dictionary

    Returns:
        results_list: List of results for each label fraction
    """
    print(f"\n{'#'*80}")
    print(f"LABEL SPARSITY EXPERIMENT - {dataset_name}")
    print(f"{'#'*80}")

    results_list = []

    for label_fraction in tqdm(
        config["label_fractions"], desc=f"{dataset_name} Label Sparsity"
    ):
        results = run_single_experiment(
            dataset_name, label_fraction, corruption_rate=0.0, config=config
        )
        results_list.append(results)

    return results_list


def run_label_noise_experiment(dataset_name, config, label_fraction=0.1):
    """
    Run experiment varying corruption rate with fixed label fraction

    Args:
        dataset_name: 'MNIST' or 'FashionMNIST'
        config: Configuration dictionary
        label_fraction: Fixed label fraction to use

    Returns:
        results_list: List of results for each corruption rate
    """
    print(f"\n{'#'*80}")
    print(
        f"LABEL NOISE EXPERIMENT - {dataset_name} (Label Fraction: {label_fraction*100:.1f}%)"
    )
    print(f"{'#'*80}")

    results_list = []

    for corruption_rate in tqdm(
        config["corruption_rates"], desc=f"{dataset_name} Noise Robustness"
    ):
        results = run_single_experiment(
            dataset_name, label_fraction, corruption_rate=corruption_rate, config=config
        )
        results_list.append(results)

    return results_list


def plot_results(sparsity_results, noise_results, dataset_name, save_path):
    """
    Plot the results of the experiments

    Args:
        sparsity_results: Results from label sparsity experiment
        noise_results: Results from label noise experiment
        dataset_name: Name of the dataset
        save_path: Path to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Label Sparsity - Accuracy
    label_fractions = [r["label_fraction"] for r in sparsity_results]
    accuracies = [r["final_test_accuracy"] for r in sparsity_results]
    axes[0, 0].plot(np.array(label_fractions) * 100, accuracies, "o-", linewidth=2)
    axes[0, 0].set_xlabel("Label Fraction (%)")
    axes[0, 0].set_ylabel("Test Accuracy")
    axes[0, 0].set_title(f"{dataset_name}: Accuracy vs Label Sparsity")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale("log")

    # Label Sparsity - Beta-VAE
    beta_scores = [r["disentanglement_metrics"]["beta_vae"] for r in sparsity_results]
    axes[0, 1].plot(
        np.array(label_fractions) * 100, beta_scores, "o-", linewidth=2, color="orange"
    )
    axes[0, 1].set_xlabel("Label Fraction (%)")
    axes[0, 1].set_ylabel("Beta-VAE Score")
    axes[0, 1].set_title(f"{dataset_name}: Beta-VAE vs Label Sparsity")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale("log")

    # Label Sparsity - Combined
    ax1 = axes[0, 2]
    ax2 = ax1.twinx()
    l1 = ax1.plot(
        np.array(label_fractions) * 100,
        accuracies,
        "o-",
        linewidth=2,
        label="Accuracy",
        color="blue",
    )
    l2 = ax2.plot(
        np.array(label_fractions) * 100,
        beta_scores,
        "s-",
        linewidth=2,
        label="Beta-VAE",
        color="red",
    )
    ax1.set_xlabel("Label Fraction (%)")
    ax1.set_ylabel("Test Accuracy", color="blue")
    ax2.set_ylabel("Beta-VAE Score", color="red")
    ax1.set_title(f"{dataset_name}: Accuracy & Disentanglement vs Sparsity")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")

    # Label Noise - Accuracy
    corruption_rates = [r["corruption_rate"] for r in noise_results]
    accuracies_noise = [r["final_test_accuracy"] for r in noise_results]
    axes[1, 0].plot(
        np.array(corruption_rates) * 100, accuracies_noise, "o-", linewidth=2
    )
    axes[1, 0].set_xlabel("Label Corruption Rate (%)")
    axes[1, 0].set_ylabel("Test Accuracy")
    axes[1, 0].set_title(f"{dataset_name}: Accuracy vs Label Noise")
    axes[1, 0].grid(True, alpha=0.3)

    # Label Noise - Beta-VAE
    beta_scores_noise = [
        r["disentanglement_metrics"]["beta_vae"] for r in noise_results
    ]
    axes[1, 1].plot(
        np.array(corruption_rates) * 100,
        beta_scores_noise,
        "o-",
        linewidth=2,
        color="orange",
    )
    axes[1, 1].set_xlabel("Label Corruption Rate (%)")
    axes[1, 1].set_ylabel("Beta-VAE Score")
    axes[1, 1].set_title(f"{dataset_name}: Beta-VAE vs Label Noise")
    axes[1, 1].grid(True, alpha=0.3)

    # Label Noise - Combined
    ax1 = axes[1, 2]
    ax2 = ax1.twinx()
    l1 = ax1.plot(
        np.array(corruption_rates) * 100,
        accuracies_noise,
        "o-",
        linewidth=2,
        label="Accuracy",
        color="blue",
    )
    l2 = ax2.plot(
        np.array(corruption_rates) * 100,
        beta_scores_noise,
        "s-",
        linewidth=2,
        label="Beta-VAE",
        color="red",
    )
    ax1.set_xlabel("Label Corruption Rate (%)")
    ax1.set_ylabel("Test Accuracy", color="blue")
    ax2.set_ylabel("Beta-VAE Score", color="red")
    ax1.set_title(f"{dataset_name}: Accuracy & Disentanglement vs Noise")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"{dataset_name}_robustness_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"Saved plot to {os.path.join(save_path, f'{dataset_name}_robustness_analysis.png')}"
    )
    plt.close()


def main(args):
    """Main function to run all experiments"""

    # Create configuration
    config = DEFAULT_CONFIG.copy()
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs
    if args.num_style:
        config["num_style"] = args.num_style
    if args.device:
        config["device"] = args.device

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
        print(f"STARTING EXPERIMENTS FOR {dataset_name}")
        print(f"{'='*80}\n")

        # Run label sparsity experiment
        if not args.noise_only:
            sparsity_results = run_label_sparsity_experiment(dataset_name, config)
            all_results[f"{dataset_name}_sparsity"] = sparsity_results

            # Save results
            results_file = os.path.join(
                config["results_path"], f"{dataset_name}_sparsity_results.json"
            )
            with open(results_file, "w") as f:
                # Convert numpy types to native Python types for JSON serialization
                serializable_results = []
                for r in sparsity_results:
                    r_copy = r.copy()
                    r_copy["train_elbos"] = [float(x) for x in r["train_elbos"]]
                    r_copy["test_elbos"] = [float(x) for x in r["test_elbos"]]
                    r_copy["test_accuracies"] = [float(x) for x in r["test_accuracies"]]
                    serializable_results.append(r_copy)
                json.dump(serializable_results, f, indent=2)
            print(f"Saved sparsity results to {results_file}")

        # Run label noise experiment
        if not args.sparsity_only:
            noise_results = run_label_noise_experiment(
                dataset_name, config, label_fraction=args.label_fraction
            )
            all_results[f"{dataset_name}_noise"] = noise_results

            # Save results
            results_file = os.path.join(
                config["results_path"], f"{dataset_name}_noise_results.json"
            )
            with open(results_file, "w") as f:
                serializable_results = []
                for r in noise_results:
                    r_copy = r.copy()
                    r_copy["train_elbos"] = [float(x) for x in r["train_elbos"]]
                    r_copy["test_elbos"] = [float(x) for x in r["test_elbos"]]
                    r_copy["test_accuracies"] = [float(x) for x in r["test_accuracies"]]
                    serializable_results.append(r_copy)
                json.dump(serializable_results, f, indent=2)
            print(f"Saved noise results to {results_file}")

        # Plot results
        if not args.sparsity_only and not args.noise_only:
            plot_results(
                sparsity_results, noise_results, dataset_name, config["results_path"]
            )

    print("\n\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved to: {config['results_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label Robustness and Sparsity Analysis for SSVAE"
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
        "--cuda",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Use CUDA if available (default: auto-detect)",
    )
    parser.add_argument(
        "--label_fraction",
        type=float,
        default=0.1,
        help="Label fraction for noise experiment (default: 0.1)",
    )
    parser.add_argument(
        "--sparsity_only",
        action="store_true",
        help="Run only label sparsity experiment",
    )
    parser.add_argument(
        "--noise_only", action="store_true", help="Run only label noise experiment"
    )

    args = parser.parse_args()

    main(args)
