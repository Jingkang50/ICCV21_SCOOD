import argparse
from functools import partial
from logging import getLogger

import torch
import torch.backends.cudnn as cudnn

from scood.data import get_dataloader
from scood.evaluation import Evaluator
from scood.metrics_logger import MetricsLogger
from scood.utils import init_object, load_yaml

logger = getLogger()


def main(args, config):
    # Initialize logger
    if args.tensorboard_dir:
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

        metrics_logger = MetricsLogger(
            args.tensorboard_dir, use_wandb=args.use_wandb, **wandb_args
        )
    else:
        metrics_logger = None

    test(
        config,
        args.data_dir,
        args.prefetch,
        args.checkpoint,
        args.ngpu,
        args.csv_path,
        metrics_logger=metrics_logger,
    )

    if metrics_logger:
        metrics_logger.close()


def test(
    config,
    data_dir,
    prefetch,
    checkpoint,
    ngpu,
    csv_path,
    metrics_logger=None,
):
    logger.info("============ Starting Evaluation ============")
    benchmark = config["id_dataset"]
    if benchmark == "cifar10":
        num_classes = 10
    elif benchmark == "cifar100":
        num_classes = 100

    # Init Datasets ############################################################
    logger.info("Initializing Datasets...")
    get_dataloader_default = partial(
        get_dataloader,
        root_dir=data_dir,
        benchmark=benchmark,
        num_classes=num_classes,
        stage="test",
        interpolation=config["interpolation"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=prefetch,
    )

    test_id_loader = get_dataloader_default(name=config["id_dataset"])

    test_ood_loader_list = []
    for name in config["ood_datasets"]:
        test_ood_loader = get_dataloader_default(name=name)
        test_ood_loader_list.append(test_ood_loader)

    # Init Network #############################################################
    logger.info("Initializing Network...")
    net = init_object(
        config["network"],
        config["network_args"],
        num_classes=num_classes,
    )
    logger.info(net)
    logger.info("Building network done.")

    if checkpoint:
        net.load_state_dict(torch.load(checkpoint), strict=False)
        logger.info(f"Loaded model checkpoint from {checkpoint}.")

    net.eval()

    if ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(ngpu)))

    if ngpu > 0:
        net.cuda()

    cudnn.benchmark = True  # fire on all cylinders

    # Init Evaluator ###########################################################
    logger.info("Performing Evaluation...")

    # Init postprocessor
    postprocessor = init_object(
        config["postprocess"],
        config["postprocess_args"],
    )

    evaluator = Evaluator(net)
    eval_metrics = evaluator.eval_ood(
        test_id_loader,
        test_ood_loader_list,
        postprocessor=postprocessor,
        method=config["eval_method"],
        dataset_type=config["dataset_type"],
        csv_path=csv_path,
    )

    # Init logger
    if metrics_logger:
        metrics_logger.write_scalar_dict(
            {f"test/{k}": v for k, v in eval_metrics.items()}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="path to config file",
        default="configs/test/cifar10.yml",
    )
    parser.add_argument(
        "--checkpoint",
        help="path to model checkpoint",
        default="output/net.ckpt",
    )
    parser.add_argument(
        "--data_dir",
        help="directory to dataset",
        default="data",
    )
    parser.add_argument(
        "--csv_path",
        help="path to save evaluation results",
        default="results.csv",
    )
    parser.add_argument(
        "--tensorboard_dir",
        help="specify tensorboard directory if want to log test results",
    )

    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--prefetch", type=int, default=4, help="pre-fetching threads.")

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

    main(args, config)
