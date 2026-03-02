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
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


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

    def forward(self, z):
        mean = self.decoder_net(z)
        std = torch.exp(self.log_std).expand_as(mean)
        return td.Independent(td.Normal(loc=mean, scale=std), 2)


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

    def elbo(self, x, prior, beta):
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

        if prior == "gaussian":
            kl = td.kl_divergence(q, self.prior())
            elbo = torch.mean(log_px_z - beta * kl, dim=0)

        else:  # covers MoG and any other prior
            log_qz_x = q.log_prob(z)
            log_pz = self.prior().log_prob(z)
            kl = log_qz_x - log_pz
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
    
    def forward(self, x, prior, beta):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x, prior, beta)


def train(model, optimizer, data_loader, epochs, device, prior, beta):
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
            optimizer.zero_grad()
            loss = model(x, prior, beta)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


def evaluate_elbo(model, data_loader, device, prior, beta):
    model.eval()
    total_elbo = 0.0
    n = 0

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            elbo = model.elbo(x, prior, beta)
            total_elbo += elbo.item() * x.size(0)
            n += x.size(0)

    return total_elbo / n


def plot_aggregate_posterior(model, data_loader, device, latent_dim):
    model.eval()

    zs = []
    ys = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            q = model.encoder(x)
            z = q.rsample()
            zs.append(z.cpu())
            ys.append(y)

    zs = torch.cat(zs, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()

    if latent_dim > 2:
        pca = PCA(n_components=2)
        zs = pca.fit_transform(zs)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(zs[:,0], zs[:,1], c=ys, cmap="tab10", s=5)
    plt.colorbar(scatter)
    plt.title("Aggregate Posterior Samples")
    plt.tight_layout()
    plt.close()


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
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian',choices=['gaussian', 'mog', 'flow', 'ddpm'], help='prior type (default: %(default)s)')
    parser.add_argument('--num-components', type=int, default=10, help='number of mixture components for MoG prior (default: %(default)s)')
    parser.add_argument('--decoder', type=str, default='bernoulli',choices=['bernoulli', 'gaussian'], help='decoder type (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=1.0, help='beta parameter for beta-VAE (default: %(default)s)')
    parser.add_argument('--mask', type=str, default='random', choices=['first_half', 'random', 'chequerboard'], help='masking strategy for flow prior (default: %(default)s)')

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

    elif args.prior == 'ddpm':
        ...

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
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device, args.prior, args.beta)

        # Save model
        torch.save(model.state_dict(), args.model)

        test_elbo = evaluate_elbo(model, mnist_test_loader, device, args.prior, args.beta)
        print("Test ELBO:", test_elbo)

        plot_aggregate_posterior(model, mnist_test_loader, device, M, args.posterior)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)

