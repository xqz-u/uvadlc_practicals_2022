import logging
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import utils as u

logger = logging.getLogger(__name__)


def savefig(fig: plt.Figure, savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.tight_layout()
    plt.savefig(savepath)
    logger.info("Saved plot to '%s'", savepath)


def parse_np_array(x: str, dtype: str = "float64") -> np.ndarray:
    return np.fromstring(x.strip("[]"), sep=",").astype(dtype)


def read_metrics(fname: str) -> pd.DataFrame:
    df = pd.read_csv(fname)
    phases = {"Train", "Validation", "Test"}
    metrics_names = {"loss", "accuracy"}
    array_columns = [f"{p.lower()}_{m}" for m in metrics_names for p in phases]
    array_columns += ["precision", "recall", "f1_beta"]
    df[array_columns] = df[array_columns].apply(lambda col: col.apply(parse_np_array))
    df["hidden_dims"] = df["hidden_dims"].apply(lambda x: parse_np_array(x, "int64"))
    return df


def plot_one_experiment(df: pd.Series, savepath: str, mlp_type: str) -> plt.Axes:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    axes[0].plot(range(len(df["train_loss"])), df["train_loss"])
    axes[0].set_ylabel("Mean Cross-Entropy loss")
    axes[0].set_title("Training loss")

    axes[1].plot(range(len(df["validation_accuracy"])), df["validation_accuracy"])
    axes[1].set_ylabel("Mean accuracy")
    axes[1].set_title("Accuracy")

    fig.suptitle(f"{mlp_type} MLP")

    for ax in axes:
        ax.set_xlabel("Epochs")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if savepath:
        savefig(fig, savepath)
    return ax


def plot_experiments(df: pd.DataFrame, labs: np.ndarray, savepath: str) -> plt.Axes:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    axes[0].plot(np.vstack(df["train_loss"]).T, label=labs)
    axes[0].set_ylabel("Mean Cross-Entropy loss")
    axes[0].set_title("Training loss")

    axes[1].plot(np.vstack(df["validation_accuracy"]).T, label=labs)
    axes[1].set_ylabel("Mean accuracy")
    axes[1].set_title("Accuracy")

    fig.suptitle("Learning curves with different learning rates")

    for ax in axes:
        ax.set_xlabel("Epochs")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()

    if savepath:
        savefig(fig, savepath)
    return ax


if __name__ == "__main__":
    data = read_metrics("data/assets/assignment_experiments.csv")
    base_model = data.iloc[5]

    for b in [0.1, 1, 10]:
        logger.info(
            f"beta: {b} f1 score: {u.f1_beta_score(base_model.precision, base_model.recall, b)}"
        )

    plot_one_experiment(base_model, "data/assets/base_plot.png", "PyTorch")

    base_conf_mat = np.load("data/assets/confmat_pytorch_mlp_exp_5.npy")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        base_conf_mat,
        annot=True,
        fmt=".0f",
        xticklabels=u.classes_names,
        yticklabels=u.classes_names,
        cmap=sns.cm.rocket_r,
        ax=ax,
    )
    ax.set_title("Confusion matrix")
    savefig(fig, "data/assets/conf_mat_sns.png")

    lr_experiments = data.iloc[list(range(9))]
    plot_experiments(
        lr_experiments, lr_experiments["lr"].to_list(), "data/assets/lr_plot.png"
    )

    hiddens_experiments = data.iloc[list(range(9, 12))]
    plot_experiments(
        hiddens_experiments,
        hiddens_experiments["hidden_dims"].to_list(),
        "data/assets/hiddens_plot.png",
    )

    numpy_data = read_metrics("data/assets/assignment_experiments_numpy.csv")
    plot_one_experiment(numpy_data.iloc[0], "data/assets/base_plot_numpy.png", "NumPy")

    # NOTE you can check that the indexed rows from `data` are the
    # ones requested in the assignments by looking at the configs file
    # import json
    # with open("data/experiments_configs.json") as fd:
    #     confs = json.load(fd)
