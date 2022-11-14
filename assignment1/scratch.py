import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils import tensorboard as tb

import cifar10_utils
import mlp_pytorch
import utils as u
from train_mlp_pytorch import *

u.setup_root_logging(logging.INFO)

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


def show_cifar10_datapoint(x: torch.Tensor, label: str) -> np.ndarray:
    lab = int(label)
    # cifar10 images have shape (3, 32, 32)
    x = np.transpose(x.numpy(), (1, 2, 0))
    # rescale float image to [0, 1] to avoid matplotlib warnings
    x = x.reshape(-1, 3)
    x = (x - x.min(0)) / (x.max(0) - x.min(0))
    x = x.reshape(32, 32, 3)
    plt.imshow(x)
    plt.title(f"Class {lab}: {classes[lab]}")
    return x


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
img = show_cifar10_datapoint(xs[i], labs[i])


mlp = mlp_pytorch.MLP(32 * 32 * 3, [128], 10, use_batch_norm=True)
# img = xs[0]
# mlp(img[None, :])
logits = mlp(xs)

mlp = mlp_pytorch.MLP(32 * 32 * 3, [128], 10)
mlp(xs)

mlp = mlp_pytorch.MLP(32 * 32 * 3, [], 10)
preds = mlp(xs)


loss = torch.nn.CrossEntropyLoss()

loss(logits, labs)


riter = tb.SummaryWriter("data/tensorboard/scratch")

# Write image data to TensorBoard log dir
# images should be CxWxH
# writer.add_image("test cifar10 image", np.transpose(img, (2, 0, 1)))
# writer.flush()

with torch.no_grad():
    preds_ng = mlp(xs)

mlp = mlp.eval()
preds_ng1 = mlp(xs)
mlp = mlp.train()


preds.requires_grad
preds_ng.requires_grad
preds_ng1.requires_grad


cifar10 = cifar10_utils.get_cifar10("data")
cifar10_loader = cifar10_utils.get_dataloader(
    cifar10, batch_size=16, return_numpy=False
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(32 * 32 * 3, [128], 10, use_batch_norm=True).to(device)
kwargs = {"hidden_dims": [128], "use_batch_norm": False}
savepath = (
    f"data/assets/best_model_3072_{'-'.join(map(str, kwargs['hidden_dims']))}"
    f"_10_batch{'1' if kwargs['use_batch_norm'] else '0'}_{time.time()}.torch"
)
model.save(savepath)
# model = MLP(
#     32 * 32 * 3,
#     [128],
#     10,
#     use_batch_norm=True,
#     saved_model_path="data/assets/test_model.torch",
# ).to(device)
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
