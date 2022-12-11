#################################################################################
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
# Date Created: 2022-11-25
################################################################################

import argparse
import datetime
import itertools as it
import json
import os
from pprint import pprint

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm, trange

import utils as u
from mnist import mnist
from models import AdversarialAE


def cl_parser() -> argparse.Namespace:
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")

    # Model hyper-parameters
    parser.add_argument(
        "--z_dim", default=8, type=int, help="Dimensionality of latent code space"
    )

    # Optimizer hyper-parameters
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size to use for training"
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=0.995,
        help="Reconstruction and adversarial mixing coefficient",
    )
    parser.add_argument(
        "--ae_lr", type=float, default=1e-3, help="Autoencoder learning rate"
    )
    parser.add_argument(
        "--d_lr", type=float, default=5e-3, help="Generator learning rate"
    )

    # Other hyper-parameters
    parser.add_argument(
        "--data_dir",
        default="../data/",
        type=str,
        help="Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--num_workers",
        default=os.cpu_count(),
        type=int,
        help="Number of workers to use in the data loaders."
        + "To have a truly deterministic run, this has to be 0.",
    )
    parser.add_argument(
        "--log_dir",
        default="AAE_logs/",
        type=str,
        help="Directory where the PyTorch Lightning logs " + "should be created.",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--ae-only",
        action="store_true",
        default=False,
        help="trains only the autoencoder and leaves the discriminator untouched",
    )
    return parser.parse_args()


def generate_and_save(
    model: nn.Module, epoch: int, summary_writer: SummaryWriter, batch_size=64
):
    """
    Function that generates and save samples from the AAE latent code.
    The generated samples images should be added to TensorBoard and,
    eventually saved inside the logging directory.
    Inputs:
        model - The AAE model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and
                saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        batch_size - Number of images to generate/sample
    """
    samples = model.sample(batch_size)
    grid = make_grid(
        samples, nrow=8, normalize=True, value_range=(-1, 1), pad_value=0.5
    )
    grid = grid.detach().cpu()
    summary_writer.add_image("samples", grid, global_step=epoch)
    save_image(
        grid,
        os.path.join(
            summary_writer.log_dir,
            "results/",
            "Prior_samples/",
            "samples_epoch_{}.png".format(epoch),
        ),
    )


def save_reconstruction(model, epoch, summary_writer, data):
    """
    Function that reconstructs a batch of data from the AAE.
    The reconstructed images alongside their actual image should be added to TensorBoard and,
    eventually saved inside the logging directory.
    Inputs:
        model - The AAE model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        data - Batch of images to reconstruct
    """
    recon_batch, _ = model(data)
    n = min(data.size(0), 8)
    comparison = torch.cat([data[:n], recon_batch.view(data.size(0), 1, 28, 28)[:n]])
    grid = make_grid(
        comparison, nrow=8, normalize=True, value_range=(-1, 1), pad_value=0.5
    )
    grid = grid.detach().cpu()
    summary_writer.add_image("reconstructions", grid, global_step=epoch)
    save_image(
        comparison.data.cpu(),
        os.path.join(
            summary_writer.log_dir,
            "results/",
            "Reconstruction/",
            "reconstruction_{}.png".format(epoch),
        ),
    )


def train_aae(
    epoch: int,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    logger_ae: u.TensorBoardLogger,
    logger_disc: u.TensorBoardLogger,
    optimizer_ae: optim.Optimizer,
    optimizer_disc: optim.Optimizer,
    ae_only=False,
    lambda_=1.0,
):
    """
    Function for training an Adversarial Autoencoder model on a
    dataset for a single epoch.
    Inputs:
        epoch - Current epoch
        model - Adversarial Autoencoder model to train
        train_loader - Data Loader for the dataset you want to train on
        optimizer_ae - The optimizer used to update the parameters of
                       the generator
        optimizer_disc - The optimizer used to update the parameters
                         of the discriminator
        lambda_ - the mixing coefficient for computing the combined
                  loss in this way:
                  lambda * reconstruction loss + (1 - lambda) * adversarial loss
                  Note that lambda should be between 0 and 1
    """
    assert 0 <= lambda_ <= 1, "Lambda should be between 0 and 1. "
    model.train()
    train_loss = 0

    if ae_only:
        disc_loss = torch.Tensor([0.0])

    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(model.device)
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Encoder-Decoder update
        optimizer_ae.zero_grad()
        recon_x, z = model(x)
        ae_loss, ae_loss_dict = model.get_loss_autoencoder(x, recon_x, z, lambda_)
        ae_loss.backward()
        optimizer_ae.step()
        #######################
        # END OF YOUR CODE    #
        #######################

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Discriminator update
        if not ae_only:
            optimizer_disc.zero_grad()
            disc_loss, disc_loss_dict = model.get_loss_discriminator(z.detach())
            disc_loss.backward()
            optimizer_disc.step()
        #######################
        # END OF YOUR CODE    #
        #######################
        train_loss += disc_loss.item() + ae_loss.item()

        # saves the reconstruction of the first batch on epochs 0, 1,
        # then every 5 epochs
        if (epoch <= 1 or epoch % 5 == 0) and batch_idx == 0:
            save_reconstruction(model, epoch, logger_ae.summary_writer, x)

    logger_ae.add_values(ae_loss_dict)
    if not ae_only:
        logger_disc.add_values(disc_loss_dict)

    print(
        "====> Epoch {} : Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader)
        )
    )


def main(args: argparse.Namespace):
    """
    Main Function for the full training loop of a AAE model.
    Makes use of a separate train function for a single epoch.
    Remember to implement the optimizers, everything else is provided.
    Inputs:
        args - Namespace object from the argument parser
    """

    if args.seed is not None:
        pl.seed_everything(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Preparation of logging directories
    experiment_dir = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    results_dir = os.path.join(experiment_dir, "results")
    sample_dir = os.path.join(results_dir, "Prior_samples")
    recon_dir = os.path.join(results_dir, "Reconstruction")

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    train_loader = mnist(
        root=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Create model
    model = AdversarialAE(z_dim=args.z_dim)
    model = model.to(device)

    # Create two separate optimizers for generator and discriminator
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # You can use the Adam optimizer for autoencoder and SGD for discriminator.
    # It is recommended to reduce the momentum (beta1) to e.g. 0.5 for
    # Adam optimizer.
    modules = model.children()
    aae_params = it.chain(next(modules).parameters(), next(modules).parameters())
    betas = (0.5, 0.999)
    optimizer_ae = optim.Adam(aae_params, lr=args.ae_lr, betas=betas)
    optimizer_disc = optim.SGD(next(modules).parameters(), args.d_lr)
    #######################
    # END OF YOUR CODE    #
    #######################

    with open(os.path.join(experiment_dir, "hparams.json"), "w") as f:
        json.dump({**vars(args), "betas": betas}, f, indent=4)

    # TensorBoard logger
    # See utils.py for details on "TensorBoardLogger" class
    summary_writer = SummaryWriter(experiment_dir)
    logger_ae = u.TensorBoardLogger(summary_writer, name="generator", avg_window=10)
    logger_disc = u.TensorBoardLogger(
        summary_writer, name="discriminator", avg_window=10
    )

    # Initial generation before training
    generate_and_save(model, 0, summary_writer, args.batch_size)

    # Training loop
    print(f"Using device {device}")
    for epoch in range(args.epochs):
        # Training epoch
        train_aae(
            epoch,
            model,
            train_loader,
            logger_ae,
            logger_disc,
            optimizer_ae,
            optimizer_disc,
            ae_only=args.ae_only,
            lambda_=args.lambda_,
        )

        # Logging images every 5 epochs
        if epoch == 0 or (epoch + 1) % 5 == 0:
            generate_and_save(model, epoch + 1, summary_writer)

        # Saving last model (only every 10 epochs to reduce IO traffic)
        # As we do not have a validation step, we cannot determine the "best"
        # checkpoint during training except looking at the samples.
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(), os.path.join(checkpoint_dir, "model_checkpoint.pt")
            )


if __name__ == "__main__":
    args = cl_parser()
    pprint(vars(args))
    main(args)

kwargs = {
    "ae_lr": 0.001,
    # "ae_only": True,
    "ae_only": False,
    "batch_size": 64,
    "d_lr": 0.005,
    "data_dir": "../data/",
    "epochs": 100,
    "lambda_": 0.995,
    "log_dir": "AAE_logs/",
    "no_cuda": False,
    "num_workers": 8,
    "seed": 42,
    "z_dim": 8,
}
args = argparse.Namespace(**kwargs)
