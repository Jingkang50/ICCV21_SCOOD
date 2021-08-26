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
    optim_args: Dict[str, Any],
    trainer_args: Dict[str, Any],
):
    trainers = {
        "oe": OETrainer,
        "udg": UDGTrainer,
    }

    return trainers[name](
        net, labeled_train_loader, unlabeled_train_loader, **optim_args, **trainer_args
    )
