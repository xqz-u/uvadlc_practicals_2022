import numpy as np
import torch
from matplotlib import pyplot as plt

import cifar10_utils
import mlp_pytorch

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def show_cifar10_datapoint(x: torch.Tensor, label: str):
    lab = int(label)
    # cifar10 images have shape (3, 32, 32)
    x = np.transpose(x.numpy(), (1, 2, 0))
    # rescale float image to [0, 1] to avoid matplotlib warnings
    x = x.reshape(-1, 3)
    x = (x - x.min(0)) / (x.max(0) - x.min(0))
    x = x.reshape(32, 32, 3)
    plt.imshow(x)
    plt.title(f"Class {lab}: {classes[lab]}")


data_dir = "data"
batch_size = 4

cifar10 = cifar10_utils.get_cifar10(data_dir)
cifar10_loader = cifar10_utils.get_dataloader(
    cifar10, batch_size=batch_size, return_numpy=False
)

trainloader = cifar10_loader["train"]
trainiter = iter(trainloader)

xs, labs = next(trainiter)
i = 3
show_cifar10_datapoint(xs[i], labs[i])


mlp = mlp_pytorch.MLP(32 * 32 * 3, [128], 10, use_batch_norm=True)
# img = xs[0]
# mlp(img[None, :])
mlp(xs)

mlp = mlp_pytorch.MLP(32 * 32 * 3, [128], 10)
mlp(xs)

mlp = mlp_pytorch.MLP(32 * 32 * 3, [], 10)
mlp(xs)
