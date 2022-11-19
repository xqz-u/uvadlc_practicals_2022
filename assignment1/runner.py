import csv
import json
import logging
import os
from typing import List

import numpy as np
from torch import multiprocessing
from torch.utils import tensorboard as tb

import train_mlp_numpy
import train_mlp_pytorch
import utils as u

logger = logging.getLogger(__name__)


def dump_metrics(metrics: List[dict], fname: str, configs: List[dict]):
    exclude_headers = {
        "data_dir",
        "assets_dir",
        "test_loss",
        "tensorboard_dir",
        "verbose",
        "accuracy",
        "epochs",
        "loss",
        "experiment_name",
    }
    phases = {"Train", "Validation", "Test"}
    metrics_names = {"loss", "accuracy"}
    get_values = lambda p, m: f"{p.lower()}_{m}"
    dump_array = lambda a: repr(a.tolist()) if isinstance(a, np.ndarray) else a
    metrics_headers = [get_values(phase, k) for phase in phases for k in metrics_names]
    fieldnames = metrics_headers.copy()
    fieldnames += [k for k in metrics[0].keys() if k not in exclude_headers]
    fieldnames += [k for k in configs[0].keys() if k not in exclude_headers]

    fname = os.path.join(configs[0]["assets_dir"], f"{fname}.csv")

    with open(fname, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for m, conf in zip(metrics, configs):
            writer.writerow(
                {
                    **{
                        get_values(phase, h): dump_array(m[h][phase])
                        for phase in phases
                        for h in metrics_names
                    },
                    **{
                        k: dump_array(v)
                        for k, v in conf.items()
                        if k not in exclude_headers
                    },
                    **{
                        k: dump_array(m[k])
                        for k in m.keys()
                        if not (k in exclude_headers or k in metrics_names)
                    },
                },
            )
    logger.info("Wrote metrics to %s", fname)


def run_experiment(model_type: str, kwargs: dict) -> u.MetricsDict:
    ext, trainer = ".torch", train_mlp_pytorch.train
    if model_type == "numpy":
        ext, trainer = ".npy", train_mlp_numpy.train

    u.setup_root_logging(logging.DEBUG if kwargs.pop("verbose") else logging.INFO)
    logger.info("Tensorboard logs folder: %s", kwargs["tensorboard_dir"])
    logger.info("Assets folder: %s", kwargs["assets_dir"])
    experiment_name = kwargs["experiment_name"]
    model, *_, info = trainer(
        **kwargs,
        num_classes=10,
        tb_writer=tb.SummaryWriter(kwargs.pop("tensorboard_dir")),
    )
    model_fname = os.path.join(
        kwargs["assets_dir"], f"best_model_{experiment_name}{ext}"
    )
    model.save(model_fname)
    logger.info("Saved best model to %s", model_fname)
    confmat_fname = os.path.join(
        kwargs["assets_dir"], f"confmat_{experiment_name}_{model_type}_.npy"
    )
    np.save(confmat_fname, info.pop("confusion_matrix"))
    logger.info("Saved test confusion matrix to %s", confmat_fname)
    return info


def run_experiments(model_type: str, configs: List[dict], outfile: str):
    nprocs = multiprocessing.cpu_count()
    if len(configs) < nprocs:
        nprocs = len(configs)
    logger.info("Number of experiments: %d Processes: %d", len(configs), nprocs)
    with multiprocessing.Pool(processes=nprocs) as p:
        ret = p.starmap(run_experiment, zip([model_type] * len(configs), configs))
    logger.info("Dumping metrics...")
    dump_metrics(ret, outfile, configs)
    return ret


# creates the expriments for the pytorch implementation
def experiments(fname: str) -> List[dict]:
    ret = []
    conf = {
        "data_dir": "data/",
        "hidden_dims": [128],
        "lr": 0.1,
        "use_batch_norm": False,
        "batch_size": 128,
        "epochs": 10,
        "seed": 42,
        "tensorboard_dir": "data/tensorboard/MLP_cifar10/pytorch",
        "assets_dir": "data/assets",
        "verbose": False,
        "experiment_name": "pytorch_mlp_exp",
    }
    # different learning rates
    for i, lr_exp in enumerate(np.arange(-6.0, 3.0)):
        c = conf.copy()
        c["lr"] = 10**lr_exp
        ret.append(c)
    # different depths, longer epochs
    for i, hiddens in enumerate([[128], [256, 128], [512, 256, 128]], start=i + 1):
        c = conf.copy()
        c["hidden_dims"] = hiddens
        c["epochs"] = 20
        ret.append(c)

    for j in range(i):
        ret[j]["experiment_name"] += f"_{j}"
        ret[j]["tensorboard_dir"] += f"/{j}"

    # dump to file to be able to recall which experiment_name corresponds to
    # which setting
    with open(fname, "w") as fd:
        json.dump(ret, fd)

    logger.info("Wrote experiment configurations to %s", fname)
    return ret


if __name__ == "__main__":
    u.setup_root_logging()
    confs = experiments("data/experiments_configs.json")

    base_conf = confs[5].copy()
    base_conf["tensorboard_dir"] = base_conf["tensorboard_dir"].replace(
        "pytorch", "numpy"
    )
    base_conf["experiment_name"] = base_conf["experiment_name"].replace(
        "pytorch", "numpy"
    )
    run_experiments("numpy", [base_conf], "assignment_experiments_numpy")

    run_experiments("pytorch", confs, "assignment_experiments")
