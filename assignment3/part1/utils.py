################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torchvision.utils import make_grid


def sample_reparameterize(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Perform the reparameterization trick to sample from a distribution with the
    given mean and std.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the
               distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes
              the standard deviation of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean
            and std. The tensor should have the same shape as the mean and std
            input tensors.
    """
    assert not (std < 0).any().item(), (
        "The reparameterization trick got a negative std as input. "
        + "Are you sure your input is std and not log_std?"
    )
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    eps = torch.randn_like(std)
    z = mean + std * eps
    #######################
    # END OF YOUR CODE    #
    #######################
    return z


def KLD(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit
    Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the
    formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the
               distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log
                  standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over
              last dimension).
              The values represent the Kullback-Leibler divergence to unit
              Gaussians.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    s = 2 * log_std
    KLD = torch.exp(s) + mean**2 - 1 - s
    KLD = KLD.sum(-1) * 0.5
    #######################
    # END OF YOUR CODE    #
    #######################
    return KLD


def elbo_to_bpd(elbo: torch.Tensor, img_shape: tuple) -> torch.Tensor:
    """
    Converts the summed negative log likelihood given by the ELBO into the bits
    per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing
                    [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given
              image.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    bpd = (elbo * np.log2(np.e)) / np.prod(img_shape[1:])
    #######################
    # END OF YOUR CODE    #
    #######################
    return bpd


def image_from_multinomial(pixels: torch.Tensor) -> torch.Tensor:
    batch_s, n_classes, w, h = pixels.shape
    pixel_probs = F.softmax(pixels, dim=1)
    pixel_probs = pixel_probs.permute(0, 2, 3, 1).reshape(-1, n_classes)
    chosen_pixel_vals = torch.multinomial(pixel_probs, 1).view(batch_s, 1, w, h).float()
    return chosen_pixel_vals


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the
    manifold should represent the decoder's output means (not binarized samples
    of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the
                    distance between different latents in percentiles is
                    1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    # Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    G = torch.distributions.Normal(0, 1)
    start, end = 0.5 / grid_size, (grid_size - 0.5) / grid_size
    # NOTE this gives percentiles spaced by 1/grid_size in the original space,
    # not in the latent one
    axis = G.icdf(torch.Tensor(np.linspace(start, end, num=grid_size)))
    coords = torch.dstack(torch.meshgrid(axis, axis, indexing="ij"))
    chosen_pixel_vals = image_from_multinomial(decoder(coords.view(-1, 2)))
    img_grid = make_grid(
        chosen_pixel_vals,
        nrow=grid_size,
        value_range=(0, 1),
        normalize=True,
        pad_value=0.5,
    )
    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid


def show(imgs, cmap=None, figsize=(10, 10)):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = tvF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), cmap=cmap)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
