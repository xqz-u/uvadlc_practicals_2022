###############################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import logging
import os
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as tvmodels

from cifar100_utils import get_test_set, get_train_validation_set

logger = logging.getLogger("Resnet18-ImageNet->Cifar100")


def setup_root_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="p%(process)s [%(asctime)s] [%(levelname)s]  %(message)s  (%(name)s:%(lineno)s)",
        datefmt="%y-%m-%d %H:%M",
    )


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloader(
    dataset: data.Dataset, batch_size: int, shuffle=True
) -> data.DataLoader:
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by
        default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    trained_weights = tvmodels.ResNet18_Weights.DEFAULT
    model = tvmodels.resnet18(weights=trained_weights)

    # Randomly initialize and modify the model's last layer for CIFAR100.
    for name, param in model.named_parameters():
        param.requires_grad = False

    model.fc = nn.Linear(512, num_classes)
    model.fc.weight = nn.Parameter(torch.normal(0.0, 0.01, (num_classes, 512)))
    model.fc.bias = nn.Parameter(torch.zeros_like(model.fc.bias))

    return model


def train_one_epoch(
    train_dataloader: data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    running_loss_period: int = 50,
    **kwargs,
) -> torch.Tensor:
    loss_module = kwargs["loss_module"]
    epoch_loss, running_loss, datapoints = 0.0, 0.0, 0
    for i, (batch, labels) in enumerate(train_dataloader):
        batch, labels = batch.to(kwargs["device"]), labels.to(kwargs["device"])
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        predictions = model(batch)
        loss = loss_module(predictions, labels)
        loss.backward()
        optimizer.step()
        datapoints += len(batch)
        epoch_loss += loss.item() * len(batch)
        running_loss += loss.item()
        if i % running_loss_period == running_loss_period - 1:
            mean_running_loss = running_loss / running_loss_period
            logger.debug(
                "[%d %d train] loss: %.3f", i + 1, datapoints, mean_running_loss
            )
            running_loss = 0.0

    return epoch_loss / datapoints


def train_model(
    model,
    lr,
    batch_size,
    epochs,
    data_dir,
    checkpoint_name,
    device,
    augmentation_name=None,
):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or
                  downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """

    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(
        data_dir, augmentation_name=augmentation_name
    )
    train_loader = make_dataloader(train_dataset, batch_size=batch_size)
    val_loader = make_dataloader(val_dataset, batch_size=batch_size)

    logger.info("~ train datapoints: %d", len(train_loader) * train_loader.batch_size)
    logger.info("~ validation datapoints: %d", len(val_loader) * val_loader.batch_size)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optim = torch.optim.Adam(model.fc.parameters(), lr=lr)
    loss_module = nn.CrossEntropyLoss()

    checkpoint_fname = os.path.join(data_dir, checkpoint_name)

    # Training loop with validation after each epoch. Save the best model.
    best_accuracy = None
    for epoch in range(epochs):
        model.train()
        logger.debug("Epoch: %s", epoch)
        train_loss = train_one_epoch(
            train_loader, model, optim, loss_module=loss_module, device=device
        )
        logger.info("[%d train     ] mean loss: %.3f", epoch, train_loss)
        val_accuracy = evaluate_model(model, val_loader, device)
        logger.info("[%d validation] mean accuaracy: %.3f", epoch, val_accuracy)
        if best_accuracy is None or val_accuracy > best_accuracy:
            logger.info("[Epoch %s] update best model: %s", epoch, checkpoint_fname)
            best_accuracy = val_accuracy
            torch.save(deepcopy(model.state_dict()), checkpoint_fname)

    # Load the best model on val accuracy and return it.
    model.load_state_dict(torch.load(checkpoint_fname))
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """

    accuracies = torch.zeros(len(data_loader))
    # Set model to evaluation mode (Remember to set it back to training mode in
    # the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    with torch.no_grad():
        for i, (batch, labels) in enumerate(data_loader):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)
            predictions = logits.argmax(1)
            accuracies[i] = (predictions == labels).sum() / predictions.sum()

    accuracy = accuracies.mean()
    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or
                  downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """

    # Set the seed for reproducibility
    set_seed(seed)
    # Set the device to use for training
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    # Load the model
    model = get_model().to(device)
    # Get the augmentation to use
    ...
    # Train the model
    best_model = train_model(
        model,
        lr,
        batch_size,
        epochs,
        data_dir,
        "resnet18_cifar100_ckpt",
        device,
        augmentation_name=augmentation_name,
    )
    # Evaluate the model on the test set
    test_dataset = get_test_set(data_dir)
    test_loader = make_dataloader(test_dataset, batch_size)
    logger.info("~ test datapoints: %d", len(test_loader) * test_loader.batch_size)
    test_accuracy = evaluate_model(best_model, test_loader, device)
    logger.info("Mean test accuracy: %.3f", test_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")
    parser.add_argument("--epochs", default=30, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=123, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR100 dataset.",
    )
    parser.add_argument(
        "--augmentation_name", default=None, type=str, help="Augmentation to use."
    )

    args = parser.parse_args()
    kwargs = vars(args)
    pprint(kwargs)
    setup_root_logging()
    main(**kwargs)


# kwargs = {}
# seed = 42
# data_dir = "./data"
# augmentation_name = None
# batch_size = 128
# lr = 0.001
# epochs = 3
