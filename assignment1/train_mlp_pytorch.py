from __future__ import absolute_import, division, print_function

import argparse
import logging
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils import tensorboard as tb
from tqdm.auto import tqdm

import cifar10_utils
from mlp_pytorch import MLP

Metrics = Dict[str, Union[float, np.ndarray, torch.Tensor]]


def cl_parser() -> dict:
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
    parser.add_argument(
        "--verbose", action="store_true", help="Print training metrics to the console"
    )
    parser.add_argument(
        "--tensorboard_dir",
        default="data/tensorboard",
        type=str,
        help="Tensorboard logs directory",
    )
    return vars(parser.parse_args())


def setup_root_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s]  %(message)s  (%(name)s:%(lineno)s)",
        datefmt="%y-%m-%d %H:%M",
    )


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    logger.debug("Set seed %s for reproducibility", seed)


def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    conf_mat = torch.zeros((predictions.shape[1],) * 2)
    pred_labels = predictions.argmax(1)
    for pred, truth in zip(pred_labels, targets):
        conf_mat[truth, pred] += 1
    return conf_mat


def confusion_matrix_to_metrics(
    confusion_matrix: torch.Tensor, beta: float = 1.0, **_
) -> Metrics:
    correct_preds = torch.diag(confusion_matrix)
    col_sums = confusion_matrix.sum(0)
    row_sums = confusion_matrix.sum(1)
    precision = correct_preds / col_sums
    recall = correct_preds / row_sums
    beta_squared = beta**2
    return {
        "accuracy": correct_preds.sum() / col_sums.sum(),
        "precision": precision,
        "recall": recall,
        "f1_beta": (1 + beta_squared)
        * precision
        * recall
        / (beta_squared * precision + recall),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: data.DataLoader,
    num_classes=10,
    **kwargs,
) -> Metrics:
    mode, device = kwargs["mode"], kwargs["device"]
    loss, datapoints = 0.0, 0
    loss_module = kwargs["loss_module"]
    conf_mat = torch.zeros((num_classes,) * 2)
    model.train(False)
    with torch.no_grad():
        for i, (xs, labels) in enumerate(dataloader):
            xs = np.transpose(xs, (0, 2, 3, 1)).to(device)
            predictions = model(xs)
            loss += loss_module(predictions, labels).item()
            conf_mat += confusion_matrix(predictions, labels)
            datapoints += len(xs)
        metrics = confusion_matrix_to_metrics(conf_mat, **kwargs)
    model.train(True)
    return {**metrics, f"{mode}_loss": loss / i}


def train_one_epoch(
    train_dataloader: data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: tb.SummaryWriter,
    **kwargs,
) -> Metrics:
    loss_module, device = kwargs["loss_module"], kwargs["device"]
    epoch_loss, train_points = 0.0, 0
    for i, (xs, labels) in enumerate(train_dataloader):
        xs = np.transpose(xs, (0, 2, 3, 1)).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        predictions = model(xs)
        loss = loss_module(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        train_points += len(xs)
        if i % 50 == 49:
            mean_loss = epoch_loss / (50 * i // 49)
            logger.debug("[%d %d train] loss: %.3f", i, train_points, mean_loss)
            writer.add_scalar(
                "train_loss", mean_loss, kwargs["epoch"] * len(train_dataloader) + i
            )
    return {"train_loss": epoch_loss / i}


def train(
    hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir, **kwargs
) -> Tuple[MLP, List[float], float, Metrics]:
    # Set the random seeds for reproducibility
    set_seeds(seed)
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

    writer = tb.SummaryWriter(kwargs["tensorboard_dir"])
    writer.add_graph(model, next(iter(cifar10_loader["train"]))[0])
    writer.flush()

    evaluation_args = {
        "num_classes": kwargs["num_classes"],
        "device": device,
        "loss_module": loss_module,
        **kwargs,
    }

    best_params: OrderedDict = None
    val_accuracies = []
    for epoch in range(epochs):
        logger.debug("Epoch: %s", epoch)
        train_metrics = train_one_epoch(
            cifar10_loader["train"],
            model,
            optimizer,
            writer,
            epoch=epoch,
            **evaluation_args,
        )
        logger.info("[%d train     ] loss: %.3f", epoch, train_metrics["train_loss"])
        train_metrics.update(
            evaluate_model(
                model,
                cifar10_loader["validation"],
                mode="validation",
                **evaluation_args,
            )
        )
        logger.info(
            "[%d validation] loss: %.3f accuracy: %.3f",
            epoch,
            train_metrics["validation_loss"],
            train_metrics["accuracy"],
        )
        accuracy = train_metrics["accuracy"]
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
    return model, val_accuracies, test_metrics["accuracy"], test_metrics


if __name__ == "__main__":
    # Command line arguments
    kwargs = cl_parser()
    setup_root_logging(logging.DEBUG if kwargs.pop("verbose") else logging.INFO)
    logger = logging.getLogger("PyTorchTrainer")
    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss
    # curve here


setup_root_logging(logging.DEBUG)
logger = logging.getLogger("PyTorchTrainer")


model, val_accuracies, test_accuracy, test_metrics = train(
    [128],
    0.1,
    True,
    128,
    10,
    42,
    "data",
    num_classes=10,
    tensorboard_dir="data/tensorboard/MLP_cifar10",
)

# cifar10 = cifar10_utils.get_cifar10("data")
# cifar10_loader = cifar10_utils.get_dataloader(
#     cifar10, batch_size=16, return_numpy=False
# )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MLP(32 * 32 * 3, [128], 10, use_batch_norm=True).to(device)
# xs, labels = next(iter(cifar10_loader["validation"]))
# preds = model(xs)
# pred_labels = preds.argmax(1)
# conf_mat = confusion_matrix(preds, labels)

# model, params = train([128], 0.1, True, 128, 2, 42, "data", num_classes=10)
# model.load_state_dict(params)
# evaluate_model(
#     model,
#     cifar10_loader["validation"],
#     num_classes=10,
#     loss_module=nn.CrossEntropyLoss(),
#     mode="validation",
#     device=device,
# )
