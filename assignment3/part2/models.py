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
# Author: Deep Learning Course | Fall 2022
# Date Created: 2022-11-20
################################################################################

import functools as ft
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# NOTE tutorial 9 uses GELU activation, I am using ReLU at the moment
class ConvEncoder(nn.Module):
    def __init__(self, z_dim: int):
        """
        Convolutional Encoder network with Convolution and Linear layers, ReLU
        activations. The output layer uses a fully connected layer to embed the
        representation to a latent code with z_dim dimension.
        Inputs:
            z_dim - Dimensionality of the latent code space.
        """
        super(ConvEncoder, self).__init__()
        # For an intial architecture, you can use the encoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one
        # specified here is sufficient for the assignment.
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        act_fn = nn.ReLU
        c_hid = 16
        self.net = nn.Sequential(
            # 32x32 => 16x16
            nn.Conv2d(1, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            # 16x16 => 8x8
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            # 8x8 => 4x4
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, z_dim),
        )
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x - Input batch of Images. Shape: [B,C,H,W]
        Outputs:
            z - Output of latent codes [B, z_dim]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        z = self.net(x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return z

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class ConvDecoder(nn.Module):
    def __init__(self, z_dim: int):
        """
        Convolutional Decoder network with linear and deconvolution layers and
        ReLU activations. The output layer uses a Tanh activation function to
        scale the output between -1 and 1.
        Inputs:
              z_dim - Dimensionality of the latent code space.
        """
        super(ConvDecoder, self).__init__()
        # For an intial architecture, you can use the decoder of Tutorial 9.
        # You can set the output padding in the first transposed
        # convolution to 0 to get 28x28 outputs.
        # Feel free to experiment with the architecture yourself, but
        # the one specified here is sufficient for the assignment.
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        act_fn = nn.ReLU
        c_hid = 16
        self.linear = nn.Sequential(nn.Linear(z_dim, 2 * 16 * c_hid), act_fn())
        self.net = nn.Sequential(
            # 4x4 => 7x7
            nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            # 7x7 => 14x14
            nn.ConvTranspose2d(
                2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            # 14x14 => 28x28
            nn.ConvTranspose2d(
                c_hid,
                1,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            nn.Tanh(),
        )
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            z - Batch of latent codes. Shape: [B,z_dim]
        Outputs:
            recon_x - Reconstructed image of shape [B,C,H,W]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        recon_x = self.linear(z)
        recon_x = recon_x.reshape(recon_x.shape[0], -1, 4, 4)
        recon_x = self.net(recon_x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return recon_x


class Discriminator(nn.Module):
    def __init__(self, z_dim: int):
        """
        Discriminator network with linear layers and LeakyReLU activations.
        Inputs:
              z_dim - Dimensionality of the latent code space.
        """
        super(Discriminator, self).__init__()
        # You are allowed to experiment with the architecture and
        # change the activation function, normalization, etc.
        # However, the default setup is sufficient to generate fine
        # images and gain full points in the assignment.
        # As a default setup, we recommend 3 linear layers (512 for
        # hidden units) with LeakyReLU activation functions (negative
        # slope 0.2).
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        h_dim = 512
        act = ft.partial(nn.LeakyReLU, negative_slope=0.2)
        self.net = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            act(),
            nn.Linear(h_dim, h_dim),
            act(),
            nn.Linear(h_dim, 1),
        )
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            z - Batch of latent codes. Shape: [B,z_dim]
        Outputs:
            preds - Predictions whether a specific latent code is fake
                    (<0) or real (>0).
                    No sigmoid should be applied on the output. Shape: [B,1]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        preds = self.net(z)
        #######################
        # END OF YOUR CODE    #
        #######################
        return preds

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class AdversarialAE(nn.Module):
    def __init__(self, z_dim=8):
        """
        Adversarial Autoencoder network with a Encoder, Decoder and
        Discriminator.
        Inputs:
              z_dim - Dimensionality of the latent code space. This is
                      the number of neurons of the code layer
        """
        super(AdversarialAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = ConvEncoder(z_dim)
        self.decoder = ConvDecoder(z_dim)
        self.discriminator = Discriminator(z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
        Outputs:
            recon_x - Reconstructed image of shape [B,C,H,W]
            z - Batch of latent codes. Shape: [B,z_dim]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        z = self.encoder(x)
        recon_x = self.decoder(z)
        #######################
        # END OF YOUR CODE    #
        #######################
        return recon_x, z

    def get_loss_autoencoder(
        self, x: torch.Tensor, recon_x: torch.Tensor, z_fake: torch.Tensor, lambda_=1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
            recon_x - Reconstructed image of shape [B,C,H,W]
            z_fake - Batch of latent codes for fake samples. Shape: [B,z_dim]
            lambda_ - The reconstruction coefficient (between 0 and 1).

        Outputs:
            recon_loss - The MSE reconstruction loss between actual
                         input and its reconstructed version.
            gen_loss - The Generator loss for fake latent codes
                       extracted from input.
            ae_loss - The combined adversarial and reconstruction loss for AAE
                      lambda_ * reconstruction loss + (1 - lambda_) *
                      adversarial loss
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # generator's loss -> 0 when discriminator thinks generator's outputs
        # are actually real
        disc_preds = self.discriminator(z_fake)
        gen_loss = F.binary_cross_entropy_with_logits(
            disc_preds, torch.ones_like(disc_preds, device=self.device)
        )
        recon_loss = F.mse_loss(recon_x, x)
        ae_loss = lambda_ * recon_loss + (1.0 - lambda_) * gen_loss
        logging_dict = {
            "gen_loss": gen_loss,
            "recon_loss": recon_loss,
            "ae_loss": ae_loss,
        }
        #######################
        # END OF YOUR CODE    #
        #######################
        return ae_loss, logging_dict

    def get_loss_discriminator(
        self, z_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Inputs:
            z_fake - Batch of latent codes for fake samples. Shape: [B,z_dim]

        Outputs:
            disc_loss - The discriminator loss for real and fake latent codes.
            logging_dict - A dictionary for logging the model
                           performance by following keys:
                disc_loss - The discriminator loss for real and fake
                            latent codes.
                loss_real - The discriminator loss for latent codes
                            sampled from the standard Gaussian prior.
                loss_fake - The discriminator loss for latent codes
                            extracted by encoder from input.
                accuracy - The accuracy of the discriminator for both
                           real and fake samples.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        batch_size = z_fake.size(0)
        ones = torch.ones(batch_size, 1, device=self.device)
        zeros = torch.zeros(batch_size, 1, device=self.device)
        z_real = torch.randn_like(z_fake, device=self.device)
        disc_out = self.discriminator(torch.vstack([z_real, z_fake]))
        disc_real, disc_fake = torch.split(disc_out, batch_size)
        loss_real = F.binary_cross_entropy_with_logits(disc_real, ones)
        loss_fake = F.binary_cross_entropy_with_logits(disc_fake, zeros)
        disc_loss = 0.5 * (loss_real + loss_fake)
        disc_preds = torch.where(disc_out <= 0.0, 0.0, 1.0)
        true_targets = torch.cat([ones, zeros])
        logging_dict = {
            "disc_loss": disc_loss,
            "loss_real": loss_real,
            "loss_fake": loss_fake,
            "accuracy": torch.mean((disc_preds == true_targets).float()),
        }
        #######################
        # END OF YOUR CODE    #
        #######################

        return disc_loss, logging_dict

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Function for sampling a new batch of random or conditioned
        images from the generator.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x - Generated images of shape [B,C,H,W]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        z = torch.randn((batch_size, self.z_dim), device=self.device)
        x = self.decoder(z)
        #######################
        # END OF YOUR CODE    #
        #######################
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return self.encoder.device
