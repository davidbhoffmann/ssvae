"""
Utility functions for SSVAE experiments
Including label corruption, data loading, and training utilities
"""

import torch
import numpy as np
from random import random
from torchvision import datasets, transforms
import os


def corrupt_labels(labels, corruption_rate=0.1, num_classes=10):
    """
    Corrupt labels by randomly flipping them to wrong classes

    Args:
        labels: Original labels (tensor or numpy array)
        corruption_rate: Fraction of labels to corrupt (0.0 to 1.0)
        num_classes: Total number of classes

    Returns:
        corrupted_labels: Labels with some randomly corrupted
        corruption_mask: Boolean mask indicating which labels were corrupted
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.array(labels)

    n_samples = len(labels_np)
    n_corrupt = int(n_samples * corruption_rate)

    # Randomly select indices to corrupt
    corrupt_indices = np.random.choice(n_samples, n_corrupt, replace=False)
    corruption_mask = np.zeros(n_samples, dtype=bool)
    corruption_mask[corrupt_indices] = True

    corrupted_labels = labels_np.copy()

    # For each corrupted index, assign a random wrong label
    for idx in corrupt_indices:
        original_label = labels_np[idx]
        # Choose from all classes except the correct one
        wrong_classes = [c for c in range(num_classes) if c != original_label]
        corrupted_labels[idx] = np.random.choice(wrong_classes)

    if isinstance(labels, torch.Tensor):
        corrupted_labels = torch.tensor(
            corrupted_labels, dtype=labels.dtype, device=labels.device
        )

    return corrupted_labels, corruption_mask


def create_label_mask(data_loader, label_fraction):
    """
    Create a mask indicating which batches should use labels

    Args:
        data_loader: PyTorch DataLoader
        label_fraction: Fraction of batches to use labels for

    Returns:
        label_mask: Dictionary mapping batch index to whether labels should be used
    """
    label_mask = {}
    for b in range(len(data_loader)):
        label_mask[b] = random() < label_fraction
    return label_mask


def get_data_loaders(
    dataset_name="MNIST",
    data_path="../data",
    batch_size=128,
    num_workers=0,
    download=True,
):
    """
    Get train and test data loaders for MNIST or FashionMNIST

    Args:
        dataset_name: 'MNIST' or 'FashionMNIST'
        data_path: Path to store/load data
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        download: Whether to download data if not present

    Returns:
        train_loader, test_loader
    """
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    if dataset_name == "MNIST":
        dataset_class = datasets.MNIST
    elif dataset_name == "FashionMNIST":
        dataset_class = datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        dataset_class(
            data_path, train=True, download=download, transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_class(
            data_path, train=False, download=download, transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def train_epoch(
    data,
    enc,
    dec,
    optimizer,
    label_mask,
    label_fraction,
    num_samples,
    num_batch,
    num_pixels,
    num_digits,
    device="cpu",
    eps=1e-9,
    corruption_rate=0.0,
    alpha=0.1,
):
    """
    Train for one epoch

    Args:
        data: DataLoader
        enc: Encoder model
        dec: Decoder model
        optimizer: Optimizer
        label_mask: Dictionary indicating which batches use labels
        label_fraction: Fraction of labeled data
        num_samples: Number of samples for Monte Carlo estimation
        num_batch: Batch size
        num_pixels: Number of input pixels
        num_digits: Number of classes
        device: Device to use ('cuda', 'mps', or 'cpu')
        eps: Small epsilon for numerical stability
        corruption_rate: Rate of label corruption
        alpha: Alpha parameter for ELBO (default: 0.1)

    Returns:
        epoch_elbo: Average ELBO for the epoch
        label_mask: Updated label mask
    """
    from ssvae_model import elbo

    epoch_elbo = 0.0
    enc.train()
    dec.train()
    N = 0

    for b, (images, labels) in enumerate(data):
        if images.size()[0] == num_batch:
            N += num_batch
            images = images.view(-1, num_pixels)

            # Convert labels to one-hot
            labels_onehot = torch.zeros(num_batch, num_digits)

            # Corrupt labels if corruption_rate > 0
            if corruption_rate > 0:
                labels, _ = corrupt_labels(labels, corruption_rate, num_digits)

            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, eps, 1 - eps)

            if device != "cpu":
                images = images.to(device)
                labels_onehot = labels_onehot.to(device)

            optimizer.zero_grad()

            # Determine if this batch uses labels
            if b not in label_mask:
                label_mask[b] = random() < label_fraction

            if label_mask[b]:
                q = enc(images, labels_onehot, num_samples=num_samples)
            else:
                q = enc(images, num_samples=num_samples)

            p = dec(images, q, num_samples=num_samples, eps=eps)
            loss = -elbo(q, p, num_samples)
            loss.backward()
            optimizer.step()

            if device != "cpu":
                loss = loss.cpu()
            epoch_elbo -= loss.item()

    return epoch_elbo / N, label_mask


def test_epoch(
    data, enc, dec, num_samples, num_batch, num_pixels, device='cpu', infer=True, alpha=0.1
):
    """
    Test for one epoch

    Args:
        data: DataLoader
        enc: Encoder model
        dec: Decoder model
        num_samples: Number of samples for Monte Carlo estimation
        num_batch: Batch size
        num_pixels: Number of input pixels
        device: Device to use ('cuda', 'mps', or 'cpu')
        infer: Whether to use inference for predictions
        alpha: Alpha parameter for ELBO (default: 0.1)

    Returns:
        epoch_elbo: Average ELBO for the epoch
        epoch_correct: Accuracy
    """
    from ssvae_model import elbo

    enc.eval()
    dec.eval()
    epoch_elbo = 0.0
    epoch_correct = 0
    N = 0

    for b, (images, labels) in enumerate(data):
        if images.size()[0] == num_batch:
            N += num_batch
            images = images.view(-1, num_pixels)
            if device != "cpu":
                images = images.to(device)

            q = enc(images, num_samples=num_samples)
            p = dec(images, q, num_samples=num_samples)
            batch_elbo = elbo(q, p, num_samples)

            if device != "cpu":
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()

            if infer:
                log_p = p.log_joint(0, 1)
                log_q = q.log_joint(0, 1)
                log_w = log_p - log_q
                w = torch.nn.functional.softmax(log_w, 0)
                y_samples = q["digits"].value
                y_expect = (w.unsqueeze(-1) * y_samples).sum(0)
                _, y_pred = y_expect.max(-1)
                if device != "cpu":
                    y_pred = y_pred.cpu()
                epoch_correct += (labels == y_pred).sum().item()
            else:
                _, y_pred = q["digits"].value.max(-1)
                if device != "cpu":
                    y_pred = y_pred.cpu()
                epoch_correct += (labels == y_pred).sum().item() / (num_samples or 1.0)

    return epoch_elbo / N, epoch_correct / N
