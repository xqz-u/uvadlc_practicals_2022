import argparse
import logging
from typing import Dict, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

Metrics = Dict[str, Union[float, np.ndarray, torch.Tensor]]
MetricsDict = Union[Metrics, Dict[str, "MetricsDict"]]


def cl_parser() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verbose", action="store_true", help="Print training metrics to the console"
    )

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
        "--tensorboard_dir",
        default="data/tensorboard",
        type=str,
        help="Tensorboard logs directory",
    )
    parser.add_argument(
        "--assets_dir",
        default="data/assets",
        type=str,
        help="Directory where to store/find static assets - plots, saved models etc.",
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
    # GPU operation have separate seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    logger.debug("Set seed %d for reproducibility", seed)


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    conf_mat = np.zeros((predictions.shape[1],) * 2)
    pred_labels = predictions.argmax(1)
    for pred, truth in zip(pred_labels, targets):
        conf_mat[truth, pred] += 1
    return conf_mat


def get_true_negatives(conf_mat: np.ndarray) -> np.ndarray:
    return np.array(
        [np.delete(np.delete(conf_mat, i, 0), i, 1).sum() for i in range(len(conf_mat))]
    )


# false positives: sum along rows
# false negatives: sum along columns
def get_false_predictions(conf_mat: np.ndarray, mode: str):
    return np.sum(conf_mat - np.diag(np.diag(conf_mat)), int(mode == "neg"))


def confusion_matrix_to_metrics(
    confusion_matrix: np.ndarray, beta: float = 1.0, **_
) -> MetricsDict:
    true_pos = np.diag(confusion_matrix)
    # true_neg = get_true_negatives(confusion_matrix)
    col_sums = confusion_matrix.sum(0)
    row_sums = confusion_matrix.sum(1)
    precision = true_pos / col_sums
    recall = true_pos / row_sums
    beta_squared = beta**2
    return {
        # "accuracy": np.mean(
        #     (true_pos + true_neg)
        #     / (
        #         true_pos
        #         + true_neg
        #         + get_false_predictions(confusion_matrix, "pos")
        #         + get_false_predictions(confusion_matrix, "neg")
        #     )
        # ),
        "accuracy": true_pos.sum() / col_sums.sum(),
        "precision": precision,
        "recall": recall,
        "f1_beta": (1 + beta_squared)
        * precision
        * recall
        / (beta_squared * precision + recall),
    }


def one_hot(n_classes: int, targets: np.ndarray) -> np.ndarray:
    return np.eye(n_classes)[targets]
