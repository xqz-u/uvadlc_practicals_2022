import logging
import os
import time

import numpy as np
from matplotlib import pyplot as plt

import utils as u

logger = logging.getLogger(__name__)


def plot_model_performance(
    mlp_type: str, accuracy: np.ndarray, loss_dict: u.MetricsDict, savepath: str = None
) -> plt.Axes:
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))

    losses = np.vstack(list(loss_dict.values()))
    axes[0].plot(losses.T, label=list(loss_dict.keys()))
    axes[0].set_ylabel("Mean Cross-Entropy loss")
    axes[0].set_title("Training loss")
    # axes[0].set_xlim(0, losses.shape[1])
    # axes[0].set_ylim(losses.min(), losses.max())

    axes[1].plot(range(len(accuracy)), accuracy, label="validation")
    axes[1].set_ylabel("Mean accuracy")
    axes[1].set_title("Accuracy")
    # axes[1].set_xlim(0, len(accuracy))
    # axes[1].set_ylim(0, 1)

    for ax in axes:
        ax.set_xlabel("Epochs")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{mlp_type} MLP")
    fig.tight_layout()

    if savepath:
        if savepath.endswith(".png"):
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
        else:
            os.makedirs(savepath, exist_ok=True)
            savepath = os.path.join(
                savepath,
                f"{mlp_type}_performance_{str(time.time()).replace('.', '')}.png",
            )
        plt.savefig(savepath)
        logger.info("Saved plot to '%s'", savepath)

    return ax
