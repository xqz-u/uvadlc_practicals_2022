################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import, division, print_function

import argparse
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

import cifar10_utils
from mlp_pytorch import MLP


def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the confusion matrix, i.e. the number of true positives, false
    positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions
                   of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size
                        [n_classes, n_classes]
    """
    conf_mat = torch.zeros((predictions.shape[1],) * 2)
    pred_labels = predictions.argmax(1)
    for pred, truth in zip(pred_labels, targets):
        conf_mat[truth, pred] += 1
    return conf_mat


def confusion_matrix_to_metrics(
    confusion_matrix: torch.Tensor, beta: float = 1.0, **_
) -> Dict[str, torch.Tensor]:
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    correct_preds = torch.diag(confusion_matrix)
    col_sums = confusion_matrix.sum(0)
    row_sums = confusion_matrix.sum(1)
    precision = correct_preds / col_sums
    recall = correct_preds / row_sums
    beta_squared = beta**2
    return {
        "accuracy": correct_preds.sum() / (col_sums + row_sums).sum(),
        "precision": precision,
        "recall": recall,
        "f1_beta": (1 + beta_squared)
        * precision
        * recall
        / (beta_squared * precision + recall),
    }


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    num_classes=10,
    **kwargs,
) -> Dict[str, float]:
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """
    conf_mat = torch.zeros((num_classes,) * 2)
    model.train(False)
    with torch.no_grad():
        for xs, labels in iter(data_loader):
            xs = np.transpose(xs, (0, 2, 3, 1)).to(kwargs["device"])
            conf_mat += confusion_matrix(model(xs), labels)
        conf_mat = confusion_matrix_to_metrics(conf_mat, **kwargs)
    model.train(True)
    return conf_mat


def train(
    hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir, **kwargs
):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=False
    )
    n_classes = 10

    model_init = lambda: MLP(
        32 * 32 * 3, hidden_dims, n_classes, use_batch_norm=use_batch_norm
    )
    model = model_init().to(device)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    evaluation_args = {"num_classes": n_classes, "device": device, **kwargs}

    running_loss = 0.0
    best_model_params: OrderedDict = None
    val_accuracies = []
    for epoch in range(epochs):
        for i, (xs, labels) in enumerate(iter(cifar10_loader["train"])):
            xs = np.transpose(xs, (0, 2, 3, 1)).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            predictions = model(xs)
            loss = loss_module(predictions, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 50 == 0:  # print every 2000 mini-batches
                print(f"[{epoch}, {i:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

        val_metrics = evaluate_model(
            model, cifar10_loader["validation"], **evaluation_args
        )
        accuracy = val_metrics["accuracy"]
        if best_model_params is None or accuracy > val_accuracies[-1]:
            print(f"Epoch {epoch}, found better params!")
            best_model_params = model.state_dict()
        val_accuracies.append(accuracy)

    print(val_accuracies)
    test_accuracy = evaluate_model(
        model_init().load_state_dict(best_model_params),
        # .to(device),
        cifar10_loader["test"],
        **evaluation_args,
    )["accuracy"]
    # TODO: Add any information you might want to save for plotting
    logging_info = ...
    return model, val_accuracies, test_accuracy, logging_info


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--hidden_dims",
        default=[128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
        help="Use this option to add Batch Normalization layers to the MLP.",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR10 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here


model, val_accuracies, test_accuracy, _ = train([128], 0.1, True, 128, 10, 42, "data")
