"""
Disentanglement metrics for evaluating latent representations
Implements Beta-VAE metric and Factor-VAE metric
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats


def compute_beta_vae_metric(
    enc,
    data_loader,
    num_samples=10000,
    num_batch=128,
    num_pixels=784,
    device="cpu",
    num_style=50,
):
    """
    Compute the Beta-VAE disentanglement metric

    This metric evaluates how well individual latent dimensions correspond to
    generative factors by training a classifier to predict which latent dimension
    was fixed when generating pairs of samples.

    Args:
        enc: Encoder model
        data_loader: DataLoader for the dataset
        num_samples: Number of samples to use for evaluation
        num_batch: Batch size
        num_pixels: Number of input pixels
        device: Device to use ('cuda', 'mps', or 'cpu')
        num_style: Number of latent style dimensions

    Returns:
        metric_score: Beta-VAE metric score (0 to 1, higher is better)
    """
    enc.eval()

    latents = []
    labels = []

    with torch.no_grad():
        for images, labs in data_loader:
            if images.size()[0] == num_batch and len(latents) * num_batch < num_samples:
                images = images.view(-1, num_pixels)
                if device != "cpu":
                    images = images.to(device)

                q = enc(images)
                z = q["styles"].value

                if device != "cpu":
                    z = z.cpu()

                latents.append(z.numpy())
                labels.append(labs.numpy())

            if len(latents) * num_batch >= num_samples:
                break

    latents = np.concatenate(latents, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]

    latents_mean = latents.mean(axis=0)
    latents_std = latents.std(axis=0) + 1e-8
    latents_normalized = (latents - latents_mean) / latents_std

    accuracies = []

    for dim in range(num_style):
        X = latents_normalized[:, dim : dim + 1]
        y = labels

        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        accuracies.append(acc)

    best_acc = max(accuracies)
    random_acc = 1.0 / 10
    metric_score = (best_acc - random_acc) / (1.0 - random_acc)

    return metric_score, accuracies


def compute_factor_vae_metric(
    enc,
    data_loader,
    num_samples=10000,
    num_batch=128,
    num_pixels=784,
    device="cpu",
    num_style=50,
):
    """
    Compute a Factor-VAE style disentanglement metric

    This metric measures the variance of latent dimensions across different
    factors (classes in this case) to assess disentanglement.

    Args:
        enc: Encoder model
        data_loader: DataLoader for the dataset
        num_samples: Number of samples to use for evaluation
        num_batch: Batch size
        num_pixels: Number of input pixels
        device: Device to use ('cuda', 'mps', or 'cpu')
        num_style: Number of latent style dimensions

    Returns:
        metric_score: Factor-VAE metric score
    """
    enc.eval()

    latents_by_class = {i: [] for i in range(10)}

    with torch.no_grad():
        total_samples = 0
        for images, labels in data_loader:
            if images.size()[0] == num_batch and total_samples < num_samples:
                images = images.view(-1, num_pixels)
                if device != "cpu":
                    images = images.to(device)

                q = enc(images)
                z = q["styles"].value

                if device != "cpu":
                    z = z.cpu()

                z_np = z.numpy()
                labels_np = labels.numpy()

                for i in range(len(labels_np)):
                    latents_by_class[labels_np[i]].append(z_np[i])

                total_samples += num_batch

            if total_samples >= num_samples:
                break

    for k in latents_by_class:
        if len(latents_by_class[k]) > 0:
            latents_by_class[k] = np.array(latents_by_class[k])

    scores = []

    for dim in range(num_style):
        within_var = []
        for k in latents_by_class:
            if len(latents_by_class[k]) > 1:
                var = np.var(latents_by_class[k][:, dim])
                within_var.append(var)

        if len(within_var) > 0:
            avg_within_var = np.mean(within_var)
            class_means = [
                np.mean(latents_by_class[k][:, dim])
                for k in latents_by_class
                if len(latents_by_class[k]) > 0
            ]
            between_var = np.var(class_means)
            if avg_within_var > 0:
                score = between_var / (avg_within_var + 1e-8)
                scores.append(score)

    metric_score = np.mean(scores) if len(scores) > 0 else 0.0

    return metric_score, scores


def compute_mutual_information_gap(
    enc,
    data_loader,
    num_samples=5000,
    num_batch=128,
    num_pixels=784,
    device="cpu",
    num_style=50,
):
    """
    Compute the Mutual Information Gap (MIG) metric

    MIG measures the difference between the top two latent dimensions
    in terms of mutual information with each factor.

    Args:
        enc: Encoder model
        data_loader: DataLoader for the dataset
        num_samples: Number of samples to use
        num_batch: Batch size
        num_pixels: Number of input pixels
        device: Device to use ('cuda', 'mps', or 'cpu')
        num_style: Number of latent style dimensions

    Returns:
        mig_score: MIG metric score
    """
    enc.eval()

    latents = []
    labels = []

    with torch.no_grad():
        for images, labs in data_loader:
            if images.size()[0] == num_batch and len(latents) * num_batch < num_samples:
                images = images.view(-1, num_pixels)
                if device != "cpu":
                    images = images.to(device)

                q = enc(images)
                z = q["styles"].value

                if device != "cpu":
                    z = z.cpu()

                latents.append(z.numpy())
                labels.append(labs.numpy())

            if len(latents) * num_batch >= num_samples:
                break

    latents = np.concatenate(latents, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]

    num_bins = 20
    latents_discrete = np.zeros_like(latents, dtype=int)
    for dim in range(num_style):
        latents_discrete[:, dim] = np.digitize(
            latents[:, dim],
            bins=np.linspace(latents[:, dim].min(), latents[:, dim].max(), num_bins),
        )

    mutual_infos = []
    for dim in range(num_style):
        mi = mutual_info(latents_discrete[:, dim], labels)
        mutual_infos.append(mi)

    mutual_infos = np.array(mutual_infos)

    sorted_mi = np.sort(mutual_infos)[::-1]
    if len(sorted_mi) >= 2:
        mig = sorted_mi[0] - sorted_mi[1]
    else:
        mig = 0.0

    return mig, mutual_infos


def mutual_info(x, y):
    """
    Compute mutual information between two discrete variables

    Args:
        x: First variable (1D array)
        y: Second variable (1D array)

    Returns:
        mi: Mutual information
    """
    contingency = np.histogram2d(x, y, bins=(len(np.unique(x)), len(np.unique(y))))[0]

    contingency = contingency / contingency.sum()

    px = contingency.sum(axis=1)
    py = contingency.sum(axis=0)

    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if contingency[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += contingency[i, j] * np.log(contingency[i, j] / (px[i] * py[j]))

    return mi


def evaluate_all_metrics(
    enc,
    data_loader,
    num_samples=5000,
    num_batch=128,
    num_pixels=784,
    device="cpu",
    num_style=50,
):
    """
    Evaluate all disentanglement metrics

    Args:
        enc: Encoder model
        data_loader: DataLoader for the dataset
        num_samples: Number of samples to use
        num_batch: Batch size
        num_pixels: Number of input pixels
        device: Device to use ('cuda', 'mps', or 'cpu')
        num_style: Number of latent style dimensions

    Returns:
        metrics: Dictionary containing all metric scores
    """
    metrics = {}

    beta_score, beta_accs = compute_beta_vae_metric(
        enc, data_loader, num_samples, num_batch, num_pixels, device, num_style
    )
    metrics["beta_vae"] = beta_score
    metrics["beta_vae_accuracies"] = beta_accs

    factor_score, factor_scores = compute_factor_vae_metric(
        enc, data_loader, num_samples, num_batch, num_pixels, device, num_style
    )
    metrics["factor_vae"] = factor_score
    metrics["factor_vae_scores"] = factor_scores

    mig_score, mutual_infos = compute_mutual_information_gap(
        enc, data_loader, num_samples, num_batch, num_pixels, device, num_style
    )
    metrics["mig"] = mig_score
    metrics["mutual_infos"] = mutual_infos

    return metrics
