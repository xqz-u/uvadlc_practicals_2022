from __future__ import absolute_import, division, print_function

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils import tensorboard as tb

import cifar10_utils
import plot as p
import utils as u
from mlp_pytorch import MLP


def evaluate_model(
    model: nn.Module,
    dataloader: data.DataLoader,
    num_classes=10,
    **kwargs,
) -> u.MetricsDict:
    loss, datapoints = 0.0, 0
    mode, loss_module = kwargs["mode"], kwargs["loss_module"]
    conf_mat = torch.zeros((num_classes,) * 2)
    model.eval()
    with torch.no_grad():
        for i, (xs, labels) in enumerate(dataloader):
            xs = np.transpose(xs, (0, 2, 3, 1)).to(model.device)
            predictions = model(xs)
            loss += loss_module(predictions, labels).item() * len(xs)
            conf_mat += u.confusion_matrix(predictions, labels)
            datapoints += len(xs)
        metrics = u.confusion_matrix_to_metrics(conf_mat, **kwargs)
    return {**metrics, f"{mode}_loss": loss / datapoints}


def train_one_epoch(
    train_dataloader: data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    running_loss_period: int = 50,
    **kwargs,
) -> u.MetricsDict:
    loss_module = kwargs["loss_module"]
    epoch_loss, running_loss, datapoints = 0.0, 0.0, 0
    for i, (xs, labels) in enumerate(train_dataloader):
        xs = np.transpose(xs, (0, 2, 3, 1)).to(model.device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        predictions = model(xs)
        loss = loss_module(predictions, labels)
        loss.backward()
        optimizer.step()
        datapoints += len(xs)
        epoch_loss += loss.item() * len(xs)
        running_loss += loss.item()
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


# TODO timing?
def train(
    hidden_dims: List[int],
    lr: float,
    use_batch_norm: bool,
    batch_size: int,
    epochs: int,
    seed: int,
    data_dir: str,
    **kwargs,
) -> Tuple[MLP, List[float], float, u.MetricsDict]:
    # Set the random seeds for reproducibility
    u.set_seeds(seed)
    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=False
    )

    model = MLP(
        32 * 32 * 3, hidden_dims, kwargs["num_classes"], use_batch_norm=use_batch_norm
    ).to(device)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    writer = kwargs.get("tb_writer")

    evaluation_args = {
        "num_classes": kwargs["num_classes"],
        "loss_module": loss_module,
        **kwargs,
    }

    best_params: List[np.ndarray] = None
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(epochs):
        logger.debug("Epoch: %s", epoch)
        train_metrics = train_one_epoch(
            cifar10_loader["train"],
            model,
            optimizer,
            epoch=epoch,
            **evaluation_args,
        )
        model.train()
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
    if writer:
        writer.add_graph(model, next(iter(cifar10_loader["train"]))[0])
        writer.flush()
    return (
        model,
        val_accuracies,
        test_metrics["accuracy"],
        {"loss": {"Train": np.array(train_losses), "Validation": np.array(val_losses)}},
    )


if __name__ == "__main__":
    # Command line arguments
    kwargs = u.cl_parser()
    u.setup_root_logging(logging.DEBUG if kwargs.pop("verbose") else logging.INFO)
    logger = logging.getLogger("PyTorchTrainer")
    logger.info(
        "Tensorboard logs folder: %s\nAssets folder: %s",
        kwargs["tensorboard_dir"],
        kwargs["assets_dir"],
    )
    model, validation_accuracies, test_accuracy, info = train(
        **kwargs,
        num_classes=10,
        tb_writer=tb.SummaryWriter(kwargs.pop("tensorboard_dir")),
    )
    # Feel free to add any additional functions, such as plotting of the loss
    # curve here
    p.plot_model_performance(
        "PyTorch",
        validation_accuracies,
        info["loss"],
        savepath=kwargs.pop("assets_dir"),
    )


# u.setup_root_logging(logging.INFO)
# logger = logging.getLogger("PyTorchTrainer")
# writer = tb.SummaryWriter("data/tensorboard/MLP_cifar10")
# model, val_accuracies, test_accuracy, info = train(
#     [128],
#     0.1,
#     True,
#     128,
#     10,
#     42,
#     "data",
#     num_classes=10,
#     # tb_writer=writer
# )
# plot_model_performance(val_accuracies, info["loss"], savepath="data/assets")
# torch.save(model.state_dict(), "data/assets/best_model.torch")
