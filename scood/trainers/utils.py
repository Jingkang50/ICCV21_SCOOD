from pathlib import Path
from typing import Any, Dict

import torch.nn as nn
from torch.utils.data import DataLoader

from .oe_trainer import OETrainer
from .udg_trainer import UDGTrainer


def get_trainer(
    name: str,
    net: nn.Module,
    labeled_train_loader: DataLoader,
    unlabeled_train_loader: DataLoader,
    test_id_loader: DataLoader,
    test_ood_loader_list: DataLoader,
    evaluator: object,
    metrics_logger: object,
    output_dir: Path,
    save_all_model: bool,
    optim_args: Dict[str, Any],
    trainer_args: Dict[str, Any],
):
    trainers = {
        "oe": OETrainer,
        "udg": UDGTrainer,
    }

    return trainers[name](
        net,
        labeled_train_loader,
        unlabeled_train_loader,
        test_id_loader,
        test_ood_loader_list,
        evaluator,
        metrics_logger,
        output_dir,
        save_all_model,
        **optim_args,
        **trainer_args,
    )
