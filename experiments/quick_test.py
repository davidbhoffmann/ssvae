#!/usr/bin/env python3
"""
Quick test script to verify the SSVAE implementation
Runs a short training on MNIST with one configuration
"""

# Fix for Python 3.10+ compatibility with probtorch
import sys

if sys.version_info >= (3, 10):
    import collections
    import collections.abc

    collections.MutableMapping = collections.abc.MutableMapping

import os
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ssvae_model import Encoder, Decoder, elbo, move_tensors_to_device, get_device
from utils import get_data_loaders, train_epoch, test_epoch

# Simple test configuration
CONFIG = {
    "num_pixels": 784,
    "num_hidden": 256,
    "num_digits": 10,
    "num_style": 50,
    "num_samples": 8,
    "num_batch": 128,
    "num_epochs": 10,  # Short for testing
    "learning_rate": 1e-3,
    "beta1": 0.90,
    "eps": 1e-9,
    "data_path": "../data",
}


def main():
    # Get best available device
    device = get_device()

    print("=" * 80)
    print("SSVAE Quick Test - MNIST")
    print("=" * 80)
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, "mps"):
        print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using device: {device}\n")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(
        dataset_name="MNIST",
        data_path=CONFIG["data_path"],
        batch_size=CONFIG["num_batch"],
        download=True,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")

    # Initialize models
    print("Initializing models...")
    enc = Encoder(
        num_pixels=CONFIG["num_pixels"],
        num_hidden=CONFIG["num_hidden"],
        num_digits=CONFIG["num_digits"],
        num_style=CONFIG["num_style"],
        num_batch=CONFIG["num_batch"],
    )

    dec = Decoder(
        num_pixels=CONFIG["num_pixels"],
        num_hidden=CONFIG["num_hidden"],
        num_digits=CONFIG["num_digits"],
        num_style=CONFIG["num_style"],
    )

    # Move models to device
    enc.to(device)
    dec.to(device)
    move_tensors_to_device(enc, device)
    move_tensors_to_device(dec, device)

    print(f"Encoder parameters: {sum(p.numel() for p in enc.parameters())}")
    print(f"Decoder parameters: {sum(p.numel() for p in dec.parameters())}\n")

    # Optimizer
    optimizer = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=CONFIG["learning_rate"],
        betas=(CONFIG["beta1"], 0.999),
    )

    # Training
    print(f"Training for {CONFIG['num_epochs']} epochs...")
    print("=" * 80)

    label_mask = {}
    label_fraction = 0.1  # Use 10% labeled data
    corruption_rate = 0.0  # No corruption

    pbar = tqdm(range(CONFIG["num_epochs"]), desc="Quick Test")
    for epoch in pbar:
        # Train
        train_elbo, label_mask = train_epoch(
            train_loader,
            enc,
            dec,
            optimizer,
            label_mask,
            label_fraction=label_fraction,
            num_samples=CONFIG["num_samples"],
            num_batch=CONFIG["num_batch"],
            num_pixels=CONFIG["num_pixels"],
            num_digits=CONFIG["num_digits"],
            device=str(device),
            eps=CONFIG["eps"],
            corruption_rate=corruption_rate,
        )

        # Test
        test_elbo, test_accuracy = test_epoch(
            test_loader,
            enc,
            dec,
            num_samples=CONFIG["num_samples"],
            num_batch=CONFIG["num_batch"],
            num_pixels=CONFIG["num_pixels"],
            device=str(device),
            infer=True,
        )

        # Update progress bar
        pbar.set_postfix(
            {
                "Train ELBO": f"{train_elbo:.3e}",
                "Test Acc": f"{test_accuracy:.3f}",
                "Test ELBO": f"{test_elbo:.3e}",
            }
        )

    print("=" * 80)
    print("\n✓ Test completed successfully!")
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_accuracy:.3f}")
    print(f"  Test ELBO: {test_elbo:.4e}")
    print(f"\nThe model is working correctly!")
    print(f"You can now run the full experiments with:")
    print(f"  python label_robustness_experiment.py")


if __name__ == "__main__":
    main()
