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

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch + 1}/{epochs}")
            progress_bar.update()





#### evaluation of test set
@torch.no_grad()
def evaluate_elbo(model, data_loader, device):
    model.eval()
    total_elbo = 0.0
    total_n = 0

    for x, _ in data_loader:
        x = x.to(device)
        batch_size = x.size(0)

        batch_elbo = model.elbo(x)
        total_elbo += batch_elbo.item() * batch_size
        total_n += batch_size

    return total_elbo / total_n  # mean elbo per data point





@torch.no_grad()
def plot_prior(model, data_loader, device, out_file="agg_posterior.png"):
    model.eval()

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    M = model.prior.M
    
    embeddings = model.prior().rsample(sample_shape = (10000,))
    # embeddings = torch.concatenate(xs)

    pca = False
    if pca:
        pca = PCA(n_components=2)
        pca.fit(embeddings.cpu())
        embeddings = pca.transform(embeddings.cpu().detach().numpy())
    else: 
        tsne = TSNE(
            n_components=2,
            perplexity=30,        # try values between 5–50
            learning_rate='auto',
            init='pca',           # good default
            random_state=42
        )

        embeddings = tsne.fit_transform(
            embeddings.cpu().detach().numpy()
        )

    # import matplotlib.pyplot as plt

    # labels = torch.concatenate(labels).cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6)
    # plt.colorbar(label='Digit Label')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('VAE Latent Space (prior)')
    # plt.savefig('embeddings_plot.png')
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()

@torch.no_grad()
def plot_aggregate_posterior(model, data_loader, device, out_file="agg_posterior.png"):
    model.eval()

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    x1s = []
    x2s = []
    labels = []
    xs = []

    for x, y in iter(data_loader):
        x = x.to(device)
        y = y.to(device) 

        labels.append(y)
        
        dists = model.encoder.forward(x)
        embeddings = dists.rsample()
        xs.append(embeddings)
    embeddings = torch.concatenate(xs)
    print(f"Embeddings in posterior {embeddings.size()=}")
    
    pca = False
    if pca:
        pca = PCA(n_components=2)
        pca.fit(embeddings.cpu())
        embeddings = pca.transform(embeddings.cpu().detach().numpy())
    else: 
        tsne = TSNE(
            n_components=2,
            perplexity=30,        # try values between 5–50
            learning_rate='auto',
            init='pca',           # good default
            random_state=42
        )

        embeddings = tsne.fit_transform(
            embeddings.cpu().detach().numpy()
        )

    print(embeddings.shape)
    # import matplotlib.pyplot as plt

    labels = torch.concatenate(labels).cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(label='Digit Label')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('VAE Latent Space')
    # plt.savefig('embeddings_plot.png')
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()



@torch.no_grad()
def save_reconstructions(model, data_loader, device, out_file="reconstructions.png", n=16):
    model.eval()

    # Grab one batch
    x, y = next(iter(data_loader))
    x = x.to(device)

    # Only use first n images
    x = x[:n]

    # Encode -> sample z -> decode
    q = model.encoder(x)
    z = q.sample()  # or q.rsample(), doesn't matter for plotting

    px = model.decoder(z)  # distribution p(x|z)
    x_recon = (px.mean > 0.5).float()  # Bernoulli mean = probs, shape (n, 28, 28)

    # Put into (N,1,28,28) for save_image
    x_img = x.unsqueeze(1)
    x_recon_img = x_recon.unsqueeze(1)

    # Stack originals over reconstructions: 2n images
    both = torch.cat([x_img, x_recon_img], dim=0)

    # Make grid: first row originals, second row reconstructions
    grid = make_grid(both, nrow=n, padding=2)
    save_image(grid, out_file)
    print(f"Saved reconstructions to: {out_file}")