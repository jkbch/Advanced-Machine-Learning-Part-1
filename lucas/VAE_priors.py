
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image, make_grid



################# A standard Gaussian prior #################

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M))
        self.std = nn.Parameter(torch.ones(self.M))

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


##############A mixture of Gaussian (MoG) prior #####################

class MixtureGaussianPrior(nn.Module):
    def __init__(self, M, K):
        super().__init__()
        self.M = M
        self.K = K

        # mixture weights (logits) and component parameters
        self.logits = nn.Parameter(torch.zeros(K))
        self.means = nn.Parameter(torch.randn(K, M) * 0.1)  # (K, M)
        self.log_stds = nn.Parameter(torch.zeros(K, M))  # (K, M)

    def forward(self):
        mixing = td.Categorical(logits=self.logits)  # categorical over K components

        components = td.Independent(
            td.Normal(loc=self.means, scale=torch.exp(self.log_stds)),
            1
        )

        return td.MixtureSameFamily(
            mixture_distribution=mixing,
            component_distribution=components
        )



############## Flow based prior ################
class GaussianBase(nn.Module):
    """
        Define a standard Gaussian
    """
    def __init__(self, D):
        super().__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(D))
        self.std = nn.Parameter(torch.ones(D))

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MaskedCouplingLayer(nn.Module):
    def __init__(self, scale_net, translation_net, mask):
        super().__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        b = self.mask
        z_masked = z * b

        s = self.scale_net(z_masked)
        t = self.translation_net(z_masked)

        x = z_masked + (1.0 - b) * (z * torch.exp(s) + t)
        log_det_J = ((1.0 - b) * s).sum(dim=1)
        return x, log_det_J

    def inverse(self, x):
        b = self.mask
        x_masked = x * b

        s = self.scale_net(x_masked)
        t = self.translation_net(x_masked)

        z = x_masked + (1.0 - b) * ((x - t) * torch.exp(-s))
        log_det_J = -((1.0 - b) * s).sum(dim=1)
        return z, log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        super().__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        sum_log_det_J = 0.0
        for T in self.transformations:
            z, log_det_J = T(z)
            sum_log_det_J = sum_log_det_J + log_det_J
        return z, sum_log_det_J

    def inverse(self, x):
        sum_log_det_J = 0.0
        for T in reversed(self.transformations):
            x, log_det_J = T.inverse(x)
            sum_log_det_J = sum_log_det_J + log_det_J
        return x, sum_log_det_J

    def log_prob(self, x):
        u, log_det = self.inverse(x)
        return self.base().log_prob(u) + log_det

    def sample(self, n):
        u = self.base().sample((n,))
        x, _ = self.forward(u)
        return x


def random_mask(D):
    m = (torch.rand(D) > 0.5).float()
    if m.sum() == 0:
        m[0] = 1.0
    if m.sum() == D:
        m[0] = 0.0
    return m


def build_mlp(D, hidden, tanh_end=False):
    layers = [
        nn.Linear(D, hidden),
        nn.ReLU(),
        nn.Linear(hidden, D),
    ]
    if tanh_end:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class FlowPrior(nn.Module):
    """
    Prior module that exposes .log_prob(z) and .sample((n,))
    like a Distribution, but is learnable (flow parameters).
    """
    def __init__(self, latent_dim, num_transformations=8, hidden=128, mask_type="random"):
        super().__init__()
        self.M = latent_dim
        base = GaussianBase(latent_dim)

        transformations = []
        for i in range(num_transformations):
            if mask_type == "random":
                mask = random_mask(latent_dim)
            else:
                mask = torch.zeros(latent_dim)
                mask[latent_dim // 2:] = 1.0
                if i % 2 == 1:
                    mask = 1.0 - mask

            scale_net = build_mlp(latent_dim, hidden, tanh_end=True)
            translation_net = build_mlp(latent_dim, hidden, tanh_end=False)
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

        self.flow = Flow(base, transformations)

    def log_prob(self, z):
        return self.flow.log_prob(z)

    def sample(self, n):
        return self.flow.sample(n)