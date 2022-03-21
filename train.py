import argparse
import shutil
from functools import partial
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import yaml

from scood.data import get_dataloader
from scood.evaluation import Evaluator
from scood.metrics_logger import MetricsLogger
from scood.utils import create_logger, load_yaml, set_seeds, init_object
from test import test


def main(args, config, test_config):
    set_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Init logger
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(log_dir)

    logger.info("============ Starting Training ============")
    logger.info("Initialized logger.")
    logger.info(yaml.dump(config, default_flow_style=False))
    logger.info(f"The experiment will be stored in {output_dir}\n")
    logger.info("")

    # Save a copy of config file in output directory
    config_path = Path(args.config)
    config_save_path = output_dir / "config.yml"
    shutil.copy(config_path, config_save_path)

    # FIXME Can be stored in dataset object instead
    benchmark = config["dataset"]["labeled"]
    if benchmark == "cifar10":
        num_classes = 10
    elif benchmark == "cifar100":
        num_classes = 100

    # Init Datasets ############################################################
    get_dataloader_default = partial(
        get_dataloader,
        root_dir=args.data_dir,
        benchmark=benchmark,
        num_classes=num_classes,
    )

    labeled_train_loader = get_dataloader_default(
        name=config["dataset"]["labeled"],
        stage="train",
        batch_size=config["dataset"]["labeled_batch_size"],
        shuffle=True,
        num_workers=args.prefetch,
    )

    if config["dataset"]["unlabeled"] == "none":
        unlabeled_train_loader = None
    else:
        unlabeled_train_loader = get_dataloader_default(
            name=config["dataset"]["unlabeled"],
            stage="train",
            batch_size=config["dataset"]["unlabeled_batch_size"],
            shuffle=True,
            num_workers=args.prefetch,
        )

    test_id_loader = get_dataloader_default(
        name=config["dataset"]["labeled"],
        stage="test",
        batch_size=config["dataset"]["test_batch_size"],
        shuffle=False,
        num_workers=args.prefetch,
    )

    test_ood_loader_list = []
    for name in config["dataset"]["test_ood"]:
        test_ood_loader = get_dataloader_default(
            name=name,
            stage="test",
            batch_size=config["dataset"]["test_batch_size"],
            shuffle=False,
            num_workers=args.prefetch,
        )
        test_ood_loader_list.append(test_ood_loader)
    logger.info("Building data done.")

    # Init Network #############################################################
    if config['trainer_args']:
        try:
            num_clusters = config["trainer_args"]["num_clusters"]
        except KeyError:
            num_clusters = 0
    else:
        num_clusters = 0

    net = init_object(
        config["network"],
        config["network_args"],
        num_classes=num_classes,
        dim_aux=num_clusters,
    )
    logger.info(net)
    logger.info("Building network done.")

    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint), strict=False)
        logger.info("Loaded model checkpoint from {args.checkpoint}.")

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()

    cudnn.benchmark = True  # fire on all cylinders

    # Prepare Training ###########################################################
    if args.use_wandb:
        wandb_args = {
            "name": args.run_id,
            "id": args.run_id,
            "group": args.group_id,
            "project": args.project_name,
            "config": config,
        }
    else:
        wandb_args = {}

    tensorboard_dir = output_dir / "tensorboard_logs"
    metrics_logger = MetricsLogger(
        tensorboard_dir, use_wandb=args.use_wandb, **wandb_args
    )
    evaluator = Evaluator(net)

    # Init Trainer #############################################################
    trainer = init_object(
        config["trainer"],
        config["trainer_args"],
        net,
        labeled_train_loader,
        unlabeled_train_loader,
        test_id_loader,
        test_ood_loader_list,
        evaluator,
        metrics_logger,
        output_dir,
        args.save_all_model,
        **config["optim_args"],
    )
    logger.info("Building trainer done.")

    trainer.train(config["optim_args"]["epochs"])

    # Perform Testing ##########################################################
    if test_config:
        test(
            test_config,
            args.data_dir,
            args.prefetch,
            str(output_dir / "best.ckpt"),
            args.ngpu,
            str(output_dir / 'results.csv'),
            metrics_logger=metrics_logger,
        )

    metrics_logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="path to config file",
        default="configs/train/cifar10.yml",
    )
    parser.add_argument(
        "--test_config",
        help="specify path to test config file, if want to perform testing after training",
    )
    parser.add_argument(
        "--checkpoint",
        help="specify path to checkpoint if loading from pre-trained model",
    )
    parser.add_argument(
        "--data_dir",
        help="directory to dataset",
        default="data",
    )
    parser.add_argument(
        "--output_dir",
        help="directory to save experiment artifacts",
        default="output/cifar10_udg",
    )
    parser.add_argument(
        "--save_all_model",
        action="store_true",
        help="whether to save all model checkpoints",
    )
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--prefetch", type=int, default=4, help="pre-fetching threads.")
    parser.add_argument(
        "--seed", type=int, default=42, help="set seed for reproducibility."
    )

    # wandb arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="whether to use wandb for logging on top of tensorboard",
    )
    parser.add_argument(
        "--project_name",
        help="name of wandb project to log to",
    )
    parser.add_argument(
        "--run_id",
        help="unique identifier for this run (used in wandb logging)",
    )
    parser.add_argument(
        "--group_id",
        help="unique group id for this run (used in wandb logging)",
    )

    args = parser.parse_args()

    # Load config file
    config = load_yaml(args.config)
    test_config = load_yaml(args.test_config) if args.test_config else None

    main(args, config, test_config)
