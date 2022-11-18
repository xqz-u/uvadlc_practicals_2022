import logging
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import runner
import utils as u

logger = logging.getLogger(__name__)


def savefig(fig: plt.Figure, savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.tight_layout()
    plt.savefig(savepath)
    logger.info("Saved plot to '%s'", savepath)


def plot_one_experiment(df: pd.Series, savepath: str) -> plt.Axes:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    axes[0].plot(range(len(df["train_loss"])), df["train_loss"])
    axes[0].set_ylabel("Mean Cross-Entropy loss")
    axes[0].set_title("Training loss")

    axes[1].plot(range(len(df["validation_accuracy"])), df["validation_accuracy"])
    axes[1].set_ylabel("Mean accuracy")
    axes[1].set_title("Accuracy")

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

    fig.suptitle(f"Learning curves with different learning rates")

    for ax in axes:
        ax.set_xlabel("Epochs")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()

    if savepath:
        savefig(fig, savepath)
    return ax


if __name__ == "__main__":
    data = runner.read_metrics("data/assets/assignment_experiments.csv")

    plot_one_experiment(data.iloc[5], "data/assets/base_plot.png")

    base_conf_mat = np.load("data/assets/confmat_pytorch_mlp_exp_5.npy")
    sns.heatmap(
        base_conf_mat,
        annot=True,
        fmt=".0f",
        xticklabels=u.classes_names,
        yticklabels=u.classes_names,
        cmap=sns.cm.rocket_r,
    )
    plt.title("Confusion matrix")
    plt.savefig("data/assets/conf_mat_sns.png")

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

    # NOTE you can check that the indexed rows from `data` are the
    # ones requested in the assignments by looking at the configs file
    # import json
    # with open("data/experiments_configs.json") as fd:
    #     confs = json.load(fd)
