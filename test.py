import argparse
from functools import partial

import torch
import torch.backends.cudnn as cudnn

from scood.data import get_dataloader
from scood.evaluation import Evaluator
from scood.networks import get_network
from scood.postprocessors import get_postprocessor
from scood.utils import load_yaml


def main(args, config):
    benchmark = config["id_dataset"]
    if benchmark == "cifar10":
        num_classes = 10
    elif benchmark == "cifar100":
        num_classes = 100

    # Init Datasets ############################################################
    print("Initializing Datasets...")
    get_dataloader_default = partial(
        get_dataloader,
        root_dir=args.data_dir,
        benchmark=benchmark,
        num_classes=num_classes,
        stage="test",
        interpolation=config["interpolation"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=args.prefetch,
    )

    test_id_loader = get_dataloader_default(name=config["id_dataset"])

    test_ood_loader_list = []
    for name in config["ood_datasets"]:
        test_ood_loader = get_dataloader_default(name=name)
        test_ood_loader_list.append(test_ood_loader)

    # Init Network #############################################################
    print("Initializing Network...")
    net = get_network(
        config["network"],
        num_classes,
        checkpoint=args.checkpoint,
    )
    net.eval()

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True  # fire on all cylinders

    # Init Evaluator ###########################################################
    print("Starting Evaluation...")
    # Init postprocessor
    postprocess_args = config["postprocess_args"] if config["postprocess_args"] else {}
    postprocessor = get_postprocessor(config["postprocess"], **postprocess_args)

    evaluator = Evaluator(net)

    evaluator.eval_ood(
        test_id_loader,
        test_ood_loader_list,
        postprocessor=postprocessor,
        method=config["eval_method"],
        dataset_type=config["dataset_type"],
        csv_path=args.csv_path,
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
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--prefetch", type=int, default=4, help="pre-fetching threads.")

    args = parser.parse_args()

    # Load config file
    config = load_yaml(args.config)

    main(args, config)
