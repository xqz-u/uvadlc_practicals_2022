import argparse
import logging
from pprint import pprint

import torch
from torchvision import models as tvmodels

import dataset
import train

logger = logging.getLogger("Resnet18Noisy")


class DummyArgs:
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None, help="path to resume from checkpoint"
    )
    args_ = parser.parse_args()
    pprint(vars(args_))

    train.setup_root_logging()
    train.set_seed(42)

    args = DummyArgs()
    args.batch_size = 128
    args.num_workers = 4
    args.dataset = "cifar100"
    args.root = "./data"
    args.test_noise = True

    preproc = tvmodels.ResNet18_Weights.DEFAULT.transforms()
    _, _, test_dataset = dataset.load_dataset(args, preproc)
    test_loader = dataset.construct_dataloader(args, test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = torch.load(args_.resume, map_location=device)

    test_accuracy = train.evaluate_model(resnet, test_loader, device)
    logger.info("Achieved test accuracy on noisy CIFAR100: %.3f", test_accuracy)
