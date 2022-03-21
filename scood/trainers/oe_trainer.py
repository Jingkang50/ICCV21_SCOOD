import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from scood.losses import soft_cross_entropy
from scood.utils import AverageMeter, MetricMeter
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer


class OETrainer(BaseTrainer):
    def __init__(
        self,
        net: nn.Module,
        labeled_train_loader: DataLoader,
        unlabeled_train_loader: DataLoader,
        test_id_loader: DataLoader,
        test_ood_loader_list: DataLoader,
        evaluator: object,
        metrics_logger: object,
        # Saving args
        output_dir: Path,
        save_all_model: bool,
        # Optim args
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        epochs: int = 100,
        # Trainer args
        lambda_oe: float = 0.5,
    ) -> None:
        super().__init__(
            net,
            labeled_train_loader,
            unlabeled_train_loader,
            test_id_loader,
            test_ood_loader_list,
            evaluator,
            metrics_logger,
            output_dir,
            save_all_model,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            epochs=epochs,
        )
        self.lambda_oe = lambda_oe

    def before_epoch(self):
        pass

    def train_epoch(self):
        self.net.train()  # enter train mode

        # Track metrics
        self.metrics = MetricMeter()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        epoch_start_time = time.time()

        train_dataiter = iter(self.labeled_train_loader)
        unlabeled_dataiter = iter(self.unlabeled_train_loader)

        for self.batch_idx in range(len(train_dataiter)):
            batch = next(train_dataiter)
            try:
                unlabeled_batch = next(unlabeled_dataiter)
            except StopIteration:
                unlabeled_dataiter = iter(self.unlabeled_train_loader)
                unlabeled_batch = next(unlabeled_dataiter)

            self.data_time.update(time.time() - epoch_start_time)

            # Perform a single batch of training
            metrics_dict = self.train_step(batch, unlabeled_batch)

            self.batch_time.update(time.time() - epoch_start_time)
            self.metrics.update(metrics_dict)

            self.train_step_log(metrics_dict)

            self.num_iters += 1
            epoch_start_time = time.time()

    def train_step(self, batch, unlabeled_batch):
        data = batch["data"].cuda()
        target = batch["label"].cuda()

        # Classification loss
        logits_classifier = self.net(data)
        loss_cls = F.cross_entropy(logits_classifier, target)

        # OE loss
        unlabeled_data = unlabeled_batch["data"].cuda()

        logits_oe = self.net(unlabeled_data)
        loss_oe = soft_cross_entropy(logits_oe, unlabeled_batch["soft_label"].cuda())

        loss_oe *= self.lambda_oe
        loss = loss_cls + loss_oe

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Record and log metrics
        metrics_dict = {}
        metrics_dict["train/loss_cls"] = loss_cls
        metrics_dict["train/loss_oe"] = loss_oe
        metrics_dict["train/loss"] = loss

        return metrics_dict
