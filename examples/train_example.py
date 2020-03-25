#!/usr/bin/env python
"""
train_example.py

Toy training example where network is trained on a small multigaussian system
and we check if the covariance of the target is reproduced.

The covariance matrix is created below as a random positive matrix, 
stored in COV_TARGET. The matrix A is a temporary variable, used to build
a positive covariance COV_TARGET=A@A.T

Should provide context to help integrate the neural networks into more
interesting case of 2D lattice configurations

"""
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from anvil.models import RealNVP
from anvil.train import shifted_kl

L = 2 # very small system
N_UNITS = L**2

np.random.seed(seed=0)
A_NP = np.array(np.random.rand(N_UNITS, N_UNITS), dtype=np.float32)
A = torch.from_numpy(A_NP)
# just make some positive definite matrix
COV_TARGET = A@A.T

INV = torch.inverse(COV_TARGET)

def target_distribution_s(phi):
    r"""Returns action S(\phi) for a stack of states \phi, shape (N_states, D).
    The action corresponds to a toy training example of a multigaussian
    distribution.
    """
    return 0.5*((phi@INV)*phi).sum(axis=-1, keepdim=True)

def split_to_flat_index(length):
    r"""Returns a 1D tensor of size length^2. If used to index a split state
    such that \phi = (\phi_a, \phi_b) then \phi -> flattened state with \phi_a
    and \phi_b interlaced in a checkerboard pattern

    You only need to use this function if the action is defined to accept the
    flattened state geometry. If the action is defined to act just on a split
    geometry then states can be used directly from the model.

    """
    checkerboard = torch.zeros((length, length), dtype=bool)
    # set even sites to 1
    checkerboard[1::2, 1::2] = True
    checkerboard[::2, ::2] = True

    # make 2d state-like matrix filled with corresponding indices in split-flat state
    splitind_like_state = torch.zeros((length, length), dtype=torch.long)
    splitind_like_state[checkerboard] = torch.arange(
        int(length**2/2),
        dtype=torch.long,
    )
    splitind_like_state[~checkerboard] = torch.arange(
        int(length**2/2),
        length**2,
        dtype=torch.long,
    )
    return splitind_like_state.flatten()

N_BATCH = 2000
INDEX = split_to_flat_index(L)

def train_multivariate(model, epochs):
    """example of training loop of model"""
    # create your optimizer and a scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500)

    # let's use tqdm to see progress
    pbar = tqdm(range(epochs), desc=f"loss: N/A")
    for i in pbar:
        # gen simple states
        z = torch.randn((N_BATCH, N_UNITS))
        phi = model.inverse_map(z)
        # phi here will be phi = (phi_a, phi_b) so need to undo transform
        # using INDEX
        target = target_distribution_s(phi[:, INDEX])
        output = model(phi)

        model.zero_grad() # get rid of stored gradients
        loss = shifted_kl(output, target)
        loss.backward() # calc gradients

        optimizer.step()
        scheduler.step(loss)

        if (i%50) == 0:
            pbar.set_description(f"loss: {loss.item()}")

def sample_multivariate(model, n_large):
    """Analyse and plot if the covariance has been reproduced"""
    with torch.no_grad(): # don't want gradients being tracked in sampling stage
        z = torch.randn((n_large, N_UNITS))
        phi = model.inverse_map(z)
        cov = np.cov(phi[:, INDEX], rowvar=False)
    # Save plots
    fig, ax = plt.subplots()
    im = ax.imshow(cov)
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    ax.set_title("Covariance matrix sampled from model.")
    fig.tight_layout()
    fig.savefig("example_output/test.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    im = ax.imshow(COV_TARGET)
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    ax.set_title("Target covariance matrix.")
    fig.tight_layout()
    fig.savefig("example_output/targ.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    im = ax.imshow(cov/np.array(COV_TARGET))
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    ax.set_title(
        "Ratio of sampled model to target covariance matrix."
    )
    fig.tight_layout()
    fig.savefig("example_output/ratio.png")
    plt.close(fig)

def main():
    """train and sample model on toy distirbution"""
    # set seed, hopefully result is reproducible
    torch.manual_seed(0)
    # define simple model, each network is single layered
    model = RealNVP(
        size_in=N_UNITS, n_affine=8, affine_hidden_shape=(16,)
    )
    epochs = 5000 # Gives a decent enough approx.
    # model needs to learn rotation and rescale
    train_multivariate(model, epochs)
    n_large = 20000 # to estimate covmat
    sample_multivariate(model, n_large)

if __name__ == "__main__":
    main()
