"""
SSVAE Model for MNIST and FashionMNIST
Based on the semi-supervised VAE architecture with probtorch
"""

import torch
import torch.nn as nn
import probtorch
from probtorch.util import expand_inputs


def get_device():
    """Get the best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class Encoder(nn.Module):
    """Encoder network for SSVAE"""

    def __init__(
        self, num_pixels=784, num_hidden=256, num_digits=10, num_style=50, num_batch=128
    ):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(nn.Linear(num_pixels, num_hidden), nn.ReLU())
        self.digit_log_weights = nn.Linear(num_hidden, num_digits)
        self.digit_temp = torch.tensor(0.66)
        self.style_mean = nn.Linear(num_hidden + num_digits, num_style)
        self.style_log_std = nn.Linear(num_hidden + num_digits, num_style)

    @expand_inputs
    def forward(self, images, labels=None, num_samples=None):
        q = probtorch.Trace()
        hiddens = self.enc_hidden(images)
        digits = q.concrete(
            logits=self.digit_log_weights(hiddens),
            temperature=self.digit_temp,
            value=labels,
            name="digits",
        )
        hiddens2 = torch.cat([digits, hiddens], -1)
        styles_mean = self.style_mean(hiddens2)
        styles_std = torch.exp(self.style_log_std(hiddens2))
        q.normal(styles_mean, styles_std, name="styles")
        return q


class Decoder(nn.Module):
    """Decoder network for SSVAE"""

    def __init__(self, num_pixels=784, num_hidden=256, num_digits=10, num_style=50):
        super(self.__class__, self).__init__()
        self.num_digits = num_digits
        self.digit_log_weights = torch.zeros(num_digits)
        self.digit_temp = 0.66
        self.style_mean = torch.zeros(num_style)
        self.style_std = torch.ones(num_style)
        self.dec_hidden = nn.Sequential(
            nn.Linear(num_style + num_digits, num_hidden), nn.ReLU()
        )
        self.dec_image = nn.Sequential(nn.Linear(num_hidden, num_pixels), nn.Sigmoid())

    def forward(self, images, q=None, num_samples=None, eps=1e-9):
        p = probtorch.Trace()
        digits = p.concrete(
            logits=self.digit_log_weights,
            temperature=self.digit_temp,
            value=q["digits"],
            name="digits",
        )
        styles = p.normal(
            self.style_mean, self.style_std, value=q["styles"], name="styles"
        )
        hiddens = self.dec_hidden(torch.cat([digits, styles], -1))
        images_mean = self.dec_image(hiddens)
        p.loss(
            lambda x_hat, x: -(
                torch.log(x_hat + eps) * x + torch.log(1 - x_hat + eps) * (1 - x)
            ).sum(-1),
            images_mean,
            images,
            name="images",
        )
        return p


def elbo(q, p, num_samples=None, alpha=0.1):
    """Compute the Evidence Lower Bound (ELBO)"""
    if num_samples is None:
        return probtorch.objectives.montecarlo.elbo(
            q, p, sample_dim=None, batch_dim=0, alpha=alpha
        )
    else:
        return probtorch.objectives.montecarlo.elbo(
            q, p, sample_dim=0, batch_dim=1, alpha=alpha
        )


def move_tensors_to_device(obj, device):
    """Move all tensor attributes to specified device"""
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.to(device))
