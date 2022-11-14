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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import, division, print_function

import logging
from typing import List, Tuple

import numpy as np
import torch
from torch.utils import data
from torch.utils import tensorboard as tb

import cifar10_utils
import mlp_numpy as mlp
import modules as m
import plot as p
import utils as u


def evaluate_model(
    model: mlp.MLP, data_loader: data.DataLoader, num_classes=10, **kwargs
):
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
    loss, datapoints = 0.0, 0
    mode, loss_module = kwargs["mode"], kwargs["loss_module"]
    conf_mat = np.zeros((num_classes,) * 2)
    for i, (xs, labels) in enumerate(data_loader):
        xs = np.transpose(xs, (0, 2, 3, 1))
        predictions = model.forward(xs)
        loss += loss_module.forward(predictions, labels) * len(xs)
        conf_mat += u.confusion_matrix(predictions, labels)
        datapoints += len(xs)
    metrics = u.confusion_matrix_to_metrics(conf_mat, **kwargs)
    return {**metrics, f"{mode}_loss": loss / datapoints}


def optimize(model: mlp.MLP, lr: float = 0.1, **_):
    for param in ["weight", "bias"]:
        for _, linear in model.layers:
            linear.params[param] -= lr * linear.grads[param]


def train_one_epoch(
    train_dataloader: data.DataLoader,
    model: mlp.MLP,
    running_loss_period: int = 50,
    **kwargs,
) -> u.MetricsDict:
    loss_module = kwargs["loss_module"]
    epoch_loss, running_loss, datapoints = 0.0, 0.0, 0
    for i, (xs, labels) in enumerate(train_dataloader):
        xs = np.transpose(xs, (0, 2, 3, 1))
        # forward
        predictions = model.forward(xs)
        loss = loss_module.forward(predictions, labels)
        # backward
        _ = model.backward(loss_module.backward(predictions, labels))
        # optimize
        optimize(model, **kwargs)
        datapoints += len(xs)
        epoch_loss += loss * len(xs)
        running_loss += loss
        if i % running_loss_period == running_loss_period - 1:
            mean_running_loss = running_loss / running_loss_period
            logger.debug(
                "[%d %d train] loss: %.3f", i + 1, datapoints, mean_running_loss
            )
            if writer := kwargs.get("tb_writer"):
                writer.add_scalar(
                    "train_loss",
                    mean_running_loss,
                    kwargs["epoch"] * len(train_dataloader) + i,
                )
            running_loss = 0.0
    return {"train_loss": epoch_loss / datapoints}


def train(
    hidden_dims: List[int],
    lr: float,
    batch_size: int,
    epochs: int,
    seed: int,
    data_dir: str,
    **kwargs,
) -> Tuple[mlp.MLP, List[float], float, u.MetricsDict]:
    """
    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=True
    )

    model = mlp.MLP(32 * 32 * 3, hidden_dims, kwargs["num_classes"])
    loss_module = m.CrossEntropyModule()
    writer = kwargs.get("tb_writer")

    evaluation_args = {
        "num_classes": kwargs["num_classes"],
        "loss_module": loss_module,
        **kwargs,
    }

    best_params = None
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(epochs):
        logger.debug("Epoch: %s", epoch)
        train_metrics = train_one_epoch(
            cifar10_loader["train"],
            model,
            epoch=epoch,
            lr=lr,
            **evaluation_args,
        )
        train_losses.append(train_metrics["train_loss"])
        logger.info("[%d train     ] mean loss: %.3f", epoch, train_losses[-1])
        train_metrics.update(
            evaluate_model(
                model,
                cifar10_loader["validation"],
                mode="validation",
                **evaluation_args,
            )
        )
        val_losses.append(train_metrics["validation_loss"])
        logger.info(
            "[%d validation] mean loss: %.3f accuracy: %.3f",
            epoch,
            val_losses[-1],
            train_metrics["accuracy"],
        )
        accuracy = train_metrics["accuracy"]
        if writer:
            writer.add_scalar("validation_accuracy", accuracy, epoch)
            writer.flush()
        if best_params is None or accuracy > val_accuracies[-1]:
            logger.debug("[Epoch %s] update best model...", epoch)
            best_params = model.state_dict()
        val_accuracies.append(accuracy)

    model.load_state_dict(best_params)
    test_metrics = evaluate_model(
        model, cifar10_loader["test"], mode="test", **evaluation_args
    )
    return (
        model,
        val_accuracies,
        test_metrics["accuracy"],
        {"loss": {"Train": np.array(train_losses), "Validation": np.array(val_losses)}},
    )


if __name__ == "__main__":
    kwargs = u.cl_parser()
    kwargs.pop("use_batch_norm")
    u.setup_root_logging(logging.DEBUG if kwargs.pop("verbose") else logging.INFO)
    logger = logging.getLogger("NumPyTrainer")
    logger.info("Tensorboard logs folder: %s", kwargs["tensorboard_dir"])
    logger.info("Assets folder: %s", kwargs["assets_dir"])
    model, validation_accuracies, test_accuracy, info = train(
        **kwargs,
        num_classes=10,
        tb_writer=tb.SummaryWriter(kwargs.pop("tensorboard_dir")),
    )
    # Feel free to add any additional functions, such as plotting of the loss
    # curve here
    p.plot_model_performance(
        "NumPy", validation_accuracies, info["loss"], savepath=kwargs.pop("assets_dir")
    )


# u.setup_root_logging(logging.DEBUG)
# logger = logging.getLogger("NumPyTrainer")
# model, val_accuracies, test_accuracy, info = train(
#     [256, 128], 0.1, 128, 10, 42, "data", num_classes=10
# )
# p.plot_model_performance("NumPy", val_accuracies, info["loss"], savepath="data/assets")

# model = mlp.MLP(32 * 32 * 3, [128, 256], 10)
# loss_module = m.CrossEntropyModule()
# # model = mlp.MLP(10, [], 10)
# # xs, labs = np.random.randint(0, 10, size=(3, 10)), np.random.randint(0, 10, size=(3,))
# data_dir = "data"
# batch_size = 128
# cifar10 = cifar10_utils.get_cifar10(data_dir)
# cifar10_loader = cifar10_utils.get_dataloader(
#     cifar10, batch_size=batch_size, return_numpy=True
# )
# xs, labs = next(iter(cifar10_loader["train"]))
# preds = model.forward(xs)
# # # preds
# model.input_layer.grads["weight"]
# loss = loss_module.forward(preds, labs)
# # # loss
# loss_grad = loss_module.backward(preds, labs)
# in_grad = model.backward(loss_grad)
# # # model.input_layer.params["weight"]
# # # model.input_layer.grads["weight"]


# optimize(model, lr=0.1)
# model.input_layer.params["weight"]
