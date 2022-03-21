import random
from collections import defaultdict
from importlib import import_module
from logging import getLogger
from typing import Any, Dict

import numpy as np
import torch
import yaml
import torch.nn as nn

logger = getLogger()


def load_yaml(path: str):
    with open(path, "r") as file:
        try:
            yaml_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_file


def init_object(
    object_path: str, object_args: Dict[str, Any], *args, **kwargs
) -> object:
    root = "scood"

    object_args = {} if object_args is None else object_args

    module_name, object_name = object_path.rsplit(".", 1)
    module = import_module(f"{root}.{module_name}")

    kwargs = {**kwargs, **object_args}

    return getattr(module, object_name)(*args, **kwargs)


def sort_array(old_array, index_array):
    sorted_array = np.ones_like(old_array)
    sorted_array[index_array] = old_array
    return sorted_array


def set_seeds(seed=1234):
    """Set seeds for reproducibility."""
    # Missing seed for dataloader
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU


def count_net_params(network: nn.Module, requires_grad=True):
    if requires_grad:
        return sum(p.numel() for p in network.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in network.parameters())

class AverageMeter:
    """Compute and store the average and current value.
    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


class MetricMeter:
    """Store the average and current value for a set of metrics.
    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError("Input to MetricMeter.update() must be a dictionary")

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append("{} {:.4f} ({:.4f})".format(name, meter.val, meter.avg))
        return self.delimiter.join(output_str)
