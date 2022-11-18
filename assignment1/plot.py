import logging
import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import runner
import utils as u

logger = logging.getLogger(__name__)


def savefig(fig: plt.Figure, savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.tight_layout()
    plt.savefig(savepath)
    logger.info("Saved plot to '%s'", savepath)


# NOTE taken from
# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_confusion_matrix(
    conf_mat: np.ndarray, ax: plt.Axes = None, savepath: str = None
) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(8, 8))
    im, cbar = heatmap(
        conf_mat,
        u.classes_names,
        u.classes_names,
        cmap="magma_r",
        ax=ax,
        cbarlabel="Correct classifications",
    )
    annotate_heatmap(im, valfmt="{x:.0f}")
    if savepath:
        savefig(fig, savepath)
    return ax


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
