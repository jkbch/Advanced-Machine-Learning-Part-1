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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
import numpy as np
import time


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
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MoGPrior(nn.Module):
    def __init__(self, M, K=10):
        """
        Mixture of Gaussians prior.
        M: latent dimension
        K: number of mixture components
        """
        super().__init__()
        self.M = M
        self.K = K
        self.means = nn.Parameter(torch.randn(K, M))
        self.log_stds = nn.Parameter(torch.zeros(K, M))
        self.logits = nn.Parameter(torch.zeros(K))

    def forward(self):
        component = td.Independent(td.Normal(self.means, torch.exp(self.log_stds)), 1)
        mixture = td.Categorical(logits=self.logits)
        return td.MixtureSameFamily(mixture, component)


class FlowPrior(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self):
        return self

    def log_prob(self, z):
        return self.flow.log_prob(z)

    def sample(self, shape=(1,)):
        return self.flow.sample(shape)
    

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
        mean, log_var = torch.chunk(self.encoder_net(x), 2, dim=-1)
        log_var = torch.clamp(log_var, min=-20.0, max=20.0)   # prevent underflow/overflow
        std = torch.exp(0.5 * log_var)
        return td.Independent(td.Normal(loc=mean, scale=std), 1)


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

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super().__init__()
        self.decoder_net = decoder_net
        self.log_std = nn.Parameter(torch.zeros(28,28))
        #self.log_std = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        mean = self.decoder_net(z)
        std = torch.exp(self.log_std).clamp(min=1e-6).expand_as(mean)
        return td.Independent(td.Normal(loc=mean, scale=std), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, beta):
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
        self.beta = beta

        self.step_count = 0
        self.warm_up = 20000

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
        z = q.rsample()
        log_px_z = self.decoder(z).log_prob(x)
        log_qz_x = q.log_prob(z)
        log_pz = self.prior().log_prob(z)
        kl = log_qz_x - log_pz
        beta = self.beta * min(1, self.step_count / self.warm_up)
        elbo = torch.mean(log_px_z - beta * kl, dim=0)

        return elbo

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


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            model.step_count += 1

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


def evaluate_elbo(model, data_loader, device):
    model.eval()
    model.step_count = model.warm_up
    total_elbo = 0.0
    n = 0

    with torch.no_grad():
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            elbo = model.elbo(x)
            total_elbo += elbo.item() * x.size(0)
            n += x.size(0)

    return total_elbo / n


def plot_prior_and_posterior(
    model,
    data_loader,
    device,
    out_file="prior_posterior.png",
    use_pca=False,
    n_prior_samples=10000,
):
    model.eval()

    # --------- Sample Prior ---------
    if isinstance(model.prior, FlowPrior):
        prior_samples = model.prior.sample(n_prior_samples)
    else:
        prior_samples = model.prior().sample(sample_shape=(n_prior_samples,))

    # --------- Collect Posterior ---------
    posterior_samples = []
    labels = []

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        dists = model.encoder(x)
        z = dists.rsample()

        posterior_samples.append(z)
        labels.append(y)

    posterior_samples = torch.cat(posterior_samples, dim=0)
    labels = torch.cat(labels, dim=0).cpu().numpy()

    print(f"{prior_samples.size()=}")
    print(f"{posterior_samples.size()=}")

    # --------- Combine for joint embedding ---------
    all_embeddings = torch.cat([prior_samples, posterior_samples], dim=0)
    all_embeddings_np = all_embeddings.cpu().detach().numpy()

    # --------- Dimensionality Reduction ---------
    if use_pca:
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(all_embeddings_np)
    else:
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=42,
        )
        embeddings_2d = reducer.fit_transform(all_embeddings_np)

    # --------- Split back ---------
    prior_2d = embeddings_2d[:n_prior_samples]
    posterior_2d = embeddings_2d[n_prior_samples:]

    plt.figure(figsize=(10, 8))

    # ---------- PRIOR AS FILLED DENSITY ----------
    # Estimate density
    kde = gaussian_kde(prior_2d.T)

    # Create grid
    xmin, ymin = prior_2d.min(axis=0) - 1
    xmax, ymax = prior_2d.max(axis=0) + 1
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]

    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(grid_coords).reshape(xx.shape)

    # Filled contour (solid color with opacity)


    # Add dummy handle for legend
    plt.scatter([], [], color="gray", alpha=0.6, label="Prior")

    # ---------- POSTERIOR POINTS ----------
    scatter = plt.scatter(
        posterior_2d[:, 0],
        posterior_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
    )
    plt.contour(
        xx,
        yy,
        density,
        levels=30#,
        #cmap="Greys",
        #alpha=0.75,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Digit Label")

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("VAE Latent Space: Prior vs Aggregate Posterior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def plot_prior_vs_posterior(model, data_loader, device, latent_dim, file):
    # Aggregate posterior
    zs_post = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            z = model.encoder(x).rsample()
            zs_post.append(z.cpu())
    zs_post = torch.cat(zs_post, dim=0).numpy()
    
    # Sample prior
    zs_prior = model.prior().sample(torch.Size([len(zs_post)])).cpu().numpy()
    
    # Reduce to 2D if needed
    if latent_dim > 2:
        pca = PCA(n_components=2)
        zs_post = pca.fit_transform(zs_post)
        zs_prior = pca.transform(zs_prior)
    
    plt.figure(figsize=(10,5))
    plt.scatter(zs_prior[:,0], zs_prior[:,1], alpha=0.3, label='Prior', color='red')
    plt.scatter(zs_post[:,0], zs_post[:,1], alpha=0.5, label='Aggregate Posterior', color='blue')
    plt.legend()
    plt.title('Prior vs Aggregate Posterior')
    plt.tight_layout()
    plt.savefig(file)
    plt.show()


def init_vae_model(args):
    # Define prior distribution
    M = args.latent_dim
    
    if args.prior == 'gaussian':
        prior = GaussianPrior(M)
    elif args.prior == 'mog':
        K = args.num_components
        prior = MoGPrior(M, K)
    elif args.prior == 'flow':
        from flow import Flow, GaussianBase, MaskedCouplingLayer

        # Define prior distribution
        D = M
        base = GaussianBase(D)

        # Define transformations
        transformations =[]
        num_transformations = 5
        num_hidden = 8

        # Make a mask that is 1 for the first half of the features and 0 for the second half
        if args.mask == 'first_half':
            mask = torch.zeros((D,))
            mask[D//2:] = 1
        elif args.mask == 'random':
            mask = torch.randint(0, 2, (D,)).float()
        elif args.mask == 'chequerboard':
            assert int(torch.sqrt(torch.tensor(D)))**2 == D, "Chequerboard mask requires D to be a perfect square"

            N = round(torch.sqrt(D))
            mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(N) for j in range(N)])
            
        for i in range(num_transformations):
            mask = (1-mask) # Flip the mask
            scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D), nn.Tanh())
            translation_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

        # Define flow model
        flow_model = Flow(base, transformations).to(args.device)
        prior = FlowPrior(flow_model)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    if args.decoder == 'bernoulli':
        decoder = BernoulliDecoder(decoder_net)
    elif args.decoder == 'gaussian':
        decoder = GaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder, args.beta).to(args.device)

    return model


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--samples-data', type=str, default='samples.pt', help='file to save samples data in (default: %(default)s)')
    parser.add_argument('--posterior', type=str, default='posterior.png', help='file to save posterior in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian',choices=['gaussian', 'mog', 'flow', 'ddpm'], help='prior type (default: %(default)s)')
    parser.add_argument('--mask', type=str, default='random', choices=['first_half', 'random', 'chequerboard'], help='masking strategy for flow prior (default: %(default)s)')
    parser.add_argument('--num-components', type=int, default=10, help='number of mixture components for MoG prior (default: %(default)s)')
    parser.add_argument('--decoder', type=str, default='bernoulli',choices=['bernoulli', 'gaussian'], help='decoder type (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=1.0, help='beta parameter for beta-VAE (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    if args.decoder == 'bernoulli':
        # Load MNIST as binarized at 'thresshold' and create data loaders
        thresshold = 0.5
        mnist_train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: (thresshold < x).float().squeeze())
                ])),
            batch_size=args.batch_size, shuffle=True)
        mnist_test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/', 
                train=False, 
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Lambda(lambda x: (thresshold < x).float().squeeze())
                ])),
            batch_size=args.batch_size, shuffle=True)
    
    elif args.decoder == 'gaussian':
        # Load MNIST and create data loaders
        mnist_train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/', 
                train=True, 
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Lambda(lambda x: x.squeeze())
                    ])),
            batch_size=args.batch_size, shuffle=True)
        mnist_test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/', 
                train=False, 
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Lambda(lambda x: x.squeeze())
                    ])),
            batch_size=args.batch_size, shuffle=True)

    model = init_vae_model(args)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

        test_elbo = evaluate_elbo(model, mnist_test_loader, device)
        print("Test ELBO:", test_elbo)

        #plot_prior_vs_posterior(model, mnist_test_loader, device, args.latent_dim, args.posterior)
        plot_prior_and_posterior(model, mnist_test_loader, device, out_file=args.posterior)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        N = 10000

        # Start timing
        start_time = time.time()

        # If using CUDA, synchronize first
        if args.device == 'cuda':
            torch.cuda.synchronize()

        # Generate samples
        with torch.no_grad():
            samples = (model.sample(64)).cpu()

        samples = samples.view(64, 1, 28, 28)

        # If decoder outputs [0,1] directly, good. Otherwise, map appropriately
        samples = torch.clamp(samples, 0.0, 1.0).view(-1, 1, 28, 28)

        if args.device == 'cuda':
            torch.cuda.synchronize()

        # End timing
        end_time = time.time()

        print(f"Sampling {N} images took {end_time - start_time:.4f} seconds")

        # Save grid of generated digits
        torch.save(samples, args.samples_data)
        save_image(samples[:100], args.samples, nrow=10)

