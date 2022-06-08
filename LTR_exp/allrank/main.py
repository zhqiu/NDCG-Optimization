from urllib.parse import urlparse

import models.losses as losses
import numpy as np
import os
import torch
from config import Config
from data.dataset_loading import load_libsvm_dataset, create_data_loaders
from models.model import make_model
from models.model_utils import get_torch_device, CustomDataParallel
from training.train_utils import fit
from utils.command_executor import execute_command
from utils.experiments import dump_experiment_result, assert_expected_metrics
from utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from utils.ltr_logging import init_logger
from utils.python_utils import dummy_context_mgr
from argparse import ArgumentParser, Namespace
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim


def parse_args() -> Namespace:
    parser = ArgumentParser("allRank")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=True)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=True)
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")
    parser.add_argument("--pretrained-model", default="", required=False, type=str, help="Path to pretrained model")
    parser.add_argument("--random-seed", required=True, type=int)

    return parser.parse_args()


def run():
    args = parse_args()

    # reproducibility
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    # pass indices and qid into loss functions is '_M' in config_file_name
    loss_func_using_M = True if '_M' in args.config_file_name else False

    create_output_dirs(paths.output_dir)

    logger = init_logger(paths.output_dir)
    logger.info(f"created paths container {paths}")

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    output_config_path = os.path.join(paths.output_dir, "used_config.json")
    execute_command("cp {} {}".format(paths.config_path, output_config_path))

    # train_ds, val_ds
    train_ds, val_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )

    train_num_queries, train_longest_query_length = train_ds.get_u_statistics()

    n_features = train_ds.shape[-1]
    assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

    # train_dl, val_dl
    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

    # gpu support
    dev = get_torch_device()
    logger.info("Model training will execute on {}".format(dev.type))

    # instantiate model
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)

    # load optimizer, loss and LR scheduler
    optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
    if loss_func_using_M:
        loss_func = getattr(losses, config.loss.name)(train_num_queries, train_longest_query_length, **config.loss.args)
    else:
        loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
    else:
        scheduler = None

    with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
        # run training
        result = fit(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=val_dl,
            config=config,
            device=dev,
            output_dir=paths.output_dir,
            tensorboard_output_path=paths.tensorboard_output_path,
            pass_indices_n_qid=loss_func_using_M,
            pretrained_model_path=args.pretrained_model,
            **asdict(config.training)
        )

    dump_experiment_result(args, config, paths.output_dir, result)

    if urlparse(args.job_dir).scheme == "gs":
        copy_local_to_gs(paths.local_base_output_path, args.job_dir)

    assert_expected_metrics(result, config.expected_metrics)


if __name__ == "__main__":
    run()
