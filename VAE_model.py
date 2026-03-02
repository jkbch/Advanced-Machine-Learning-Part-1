# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image, make_grid
from pathlib import Path

### Import priors ###
from VAE_priors import *
from modes import *


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std_log = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std_log)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        if isinstance(prior, GaussianPrior):
            self.prior_type = "gaussian"
        elif isinstance(prior, MixtureGaussianPrior):
            self.prior_type = "mog"
        elif isinstance(prior, FlowPrior):
            self.prior_type = "flow"

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample() #used for the reparameterization trick

        if  self.prior_type == "gaussian":
            elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
            return elbo
        elif self.prior_type == "mog":
            # Reconstruction term: log p(x/z)
            log_px_given_z = self.decoder(z).log_prob(x)  # shape (B,)

            # Prior term: log p(z)  (MoG prior will compute this via logsumexp internally)
            log_pz = self.prior().log_prob(z)  # shape (B,)

            # Entropy term: log q(z|x)
            log_qz_given_x = q.log_prob(z)  # shape (B,)

            # Monte Carlo estimate of ELBO, averaged over batch
            elbo = (log_px_given_z + log_pz - log_qz_given_x).mean()
            return elbo
        else:
            log_lik = self.prior.flow.log_prob(z)
            return log_lik


    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)





if __name__ == "__main__":
    from torchvision import datasets, transforms

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', type=str, default='train',
        choices=['train', 'sample', 'eval', 'plot', 'recon'],
        help='what to do when running the script'
    )

    parser.add_argument('--model', type=str, default='model.pt',
                        help='file to save model to or load model from')
    parser.add_argument('--samples', type=str, default='samples.png',
                        help='file to save prior samples to')
    parser.add_argument('--recon', type=str, default='reconstructions.png',
                        help='file to save reconstructions to')
    parser.add_argument('--posterior-plot', type=str, default='agg_posterior.png',
                        help='file to save aggregate posterior plot to')

    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='torch device')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--latent-dim', type=int, default=32)

    # choose prior type and K for MoG
    parser.add_argument('--prior', type=str, default='gaussian',
                        choices=['gaussian', 'mog', "flow"],
                        help='prior type')
    parser.add_argument('--num-components', type=int, default=10,
                        help='K = number of mixture components for MoG prior')

    args = parser.parse_args()

    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = torch.device(args.device)

    # Load MNIST as binarized at 'thresshold' and create data loaders
    threshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                                                  transforms.Lambda(
                                                                                                      lambda x: (
                                                                                                                  threshold < x).float().squeeze())])),
                                                     batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                                                 transforms.Lambda(
                                                                                                     lambda x: (
                                                                                                                 threshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)


    # Define prior distribution
    M = args.latent_dim
    if args.prior == 'gaussian':
        prior = GaussianPrior(M)
        prior_type = 'gaussian'
    elif args.prior == 'mog':
        prior = MixtureGaussianPrior(M, args.num_components)
        prior_type = 'mog'
    elif args.prior == 'flow':
        prior = FlowPrior(latent_dim=M)
        prior_type = 'flow'
    else:
        raise ValueError("Please input recognized prior")


    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M * 2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Build model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    path_to_report = Path('..', 'report', 'ims')
    model_name = f'model_{args.prior}_latent_dim_{args.latent_dim}'
    folder = path_to_report / model_name
    folder.mkdir(exist_ok=True)
    # Modes
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, mnist_train_loader, args.epochs, device)
        torch.save(model.state_dict(), args.model)
        print(f"Saved model to: {args.model}")

        # Optional: print test ELBO after training
        test_elbo = evaluate_elbo(model, mnist_test_loader, device)
        print(f"Test ELBO (mean per datapoint): {test_elbo:.4f}")

    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=device))
        test_elbo = evaluate_elbo(model, mnist_test_loader, device)
        print(f"Test ELBO (mean per datapoint): {test_elbo:.4f}")

    elif args.mode == 'plot':
        model.load_state_dict(torch.load(args.model, map_location=device))
        plot_aggregate_posterior(model, mnist_test_loader, device, out_file=folder / 'agg_post_.png')
        plot_prior(model, mnist_test_loader, device, out_file=folder / 'prior.png')
        print(f"Saved aggregate posterior plot to: {args.posterior_plot}")

    elif args.mode == 'recon':
        model.load_state_dict(torch.load(args.model, map_location=device))
        save_reconstructions(model, mnist_test_loader, device, out_file=folder / 'recon.png', n=16)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64).cpu()
            save_image(samples.view(64, 1, 28, 28), folder / 'samples.png')
        print(f"Saved samples to: {args.samples}")