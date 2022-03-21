import datetime
import time
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scood.utils import AverageMeter, MetricMeter
from torch.utils.data import DataLoader

logger = getLogger()


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class BaseTrainer:
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
    ) -> None:
        self.net = net
        self.labeled_train_loader = labeled_train_loader
        self.unlabeled_train_loader = unlabeled_train_loader  # not utilized
        self.test_id_loader = test_id_loader
        self.test_ood_loader_list = test_ood_loader_list
        self.evaluator = evaluator
        self.metrics_logger = metrics_logger

        self.output_dir = output_dir
        self.save_all_model = save_all_model

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                epochs * len(labeled_train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / learning_rate,
            ),
        )

    def train(self, num_epochs):
        self.num_epochs = num_epochs
        self.num_iters = 0
        self.num_batches = len(iter(self.labeled_train_loader))

        self.best_metric = 0.0

        for self.epoch in range(num_epochs):
            logger.info(
                f"============ Starting epoch {self.epoch + 1} ... ============"
            )
            self.before_epoch()
            self.train_epoch()

            curr_metric = self.eval()

            self.save_model(curr_metric)

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

        for self.batch_idx in range(len(train_dataiter)):
            batch = next(train_dataiter)

            self.data_time.update(time.time() - epoch_start_time)

            # Perform a single batch of training
            metrics_dict = self.train_step(batch)

            self.batch_time.update(time.time() - epoch_start_time)
            self.metrics.update(metrics_dict)

            self.train_step_log(metrics_dict)

            self.num_iters += 1
            epoch_start_time = time.time()

    def train_step(self, batch):
        data = batch["data"].cuda()
        target = batch["label"].cuda()
        # forward
        logits_classifier = self.net(data)
        loss = F.cross_entropy(logits_classifier, target)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Record and log metrics
        metrics_dict = {}
        metrics_dict["train/loss"] = loss

        return metrics_dict

    def train_step_log(self, metrics_dict):
        # FIXME To log:
        # learning rate, sample training images, number of params
        self.metrics_logger.write_scalar_dict(metrics_dict, self.num_iters)

        if self.batch_idx % 50 == 0:
            nb_this_epoch = self.num_batches - (self.batch_idx + 1)
            nb_future_epochs = (self.num_epochs - (self.epoch + 1)) * self.num_batches
            eta_seconds = self.batch_time.avg * (nb_this_epoch + nb_future_epochs)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                "epoch [{0}/{1}][{2}/{3}]\t"
                "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "eta {eta}\t"
                "{metrics}\t".format(
                    self.epoch + 1,
                    self.num_epochs,
                    self.batch_idx + 1,
                    self.num_batches,
                    batch_time=self.batch_time,
                    data_time=self.data_time,
                    eta=eta,
                    metrics=self.metrics,
                )
            )

    def eval(self):
        classification_metrics = self.evaluator.eval_classification(self.test_id_loader)
        eval_metrics = self.evaluator.eval_ood(
            self.test_id_loader,
            self.test_ood_loader_list,
            method="full",
            dataset_type="scood",
        )

        # Log metrics
        self.metrics_logger.write_scalar_dict(
            {f"val/class/{k}": v for k, v in classification_metrics.items()},
            self.num_iters,
        )
        self.metrics_logger.write_scalar_dict(
            {f"val/oe/{k}": v for k, v in eval_metrics.items()}, self.num_iters
        )

        return classification_metrics["accuracy"]

    def save_model(self, curr_metric):
        # Save model
        torch.save(self.net.state_dict(), self.output_dir / f"epoch_{self.epoch}.ckpt")
        if not self.save_all_model:
            # Let us not waste space and delete the previous model
            prev_path = self.output_dir / f"epoch_{self.epoch - 1}.ckpt"
            prev_path.unlink(missing_ok=True)

        # Save best result
        if curr_metric >= self.best_metric:
            torch.save(self.net.state_dict(), self.output_dir / "best.ckpt")

            self.best_metric = curr_metric
