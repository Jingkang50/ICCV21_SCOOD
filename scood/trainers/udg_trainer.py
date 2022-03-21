import datetime
import time
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scood.k_means import KMeans
from scood.losses import rew_ce, rew_sce
from scood.utils import sort_array
from torch.utils.data import DataLoader

from .oe_trainer import OETrainer

logger = getLogger()


class UDGTrainer(OETrainer):
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
        num_clusters: int = 1000,
        pca_dim: int = 256,
        idf_method: str = "udg",
        purity_ind_thresh: float = 0.8,
        purity_ood_thresh: float = 0.8,
        oe_enhance_ratio: float = 2.0,
        lambda_oe: float = 0.5,
        lambda_aux: float = 0.1,
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
            lambda_oe=lambda_oe,
        )
        self.num_clusters = num_clusters
        self.idf_method = idf_method
        self.purity_ind_thresh = purity_ind_thresh
        self.purity_ood_thresh = purity_ood_thresh
        self.oe_enhance_ratio = oe_enhance_ratio
        self.lambda_aux = lambda_aux

        # Init clustering algorithm
        self.k_means = KMeans(k=num_clusters, pca_dim=pca_dim)

    def before_epoch(self):
        # Run clustering
        # FIXME Why eval mode?
        self.net.eval()

        start_time = time.time()
        # get data from train loader
        logger.info("Clustering: starting gather training features...")
        # gather train image feature
        train_idx_list, unlabeled_idx_list, feature_list, train_label_list = (
            [],
            [],
            [],
            [],
        )
        train_dataiter = iter(self.labeled_train_loader)
        for step in range(1, len(train_dataiter) + 1):
            batch = next(train_dataiter)
            index = batch["index"]
            label = batch["label"]
            # we use no augmented image for clustering
            data = batch["plain_data"].cuda()
            _, feature = self.net(data, return_feature=True)
            feature = feature.detach()
            # evaluation
            for idx in range(len(data)):
                train_idx_list.append(index[idx].tolist())
                train_label_list.append(label[idx].tolist())
                feature_list.append(feature[idx].cpu().tolist())
        num_train_data = len(feature_list)
        train_idx_list = np.array(train_idx_list, dtype=int)
        train_label_list = np.array(train_label_list, dtype=int)
        train_label_list = sort_array(train_label_list, train_idx_list)
        # in-distribution samples always have pseudo labels == actual labels
        self.labeled_train_loader.dataset.pseudo_label = train_label_list

        torch.cuda.empty_cache()

        # gather unlabeled image feature in order
        unlabeled_conf_list, unlabeled_pseudo_list = [], []
        unlabeled_dataiter = iter(self.unlabeled_train_loader)
        for step in range(1, len(unlabeled_dataiter) + 1):
            batch = next(unlabeled_dataiter)
            index = batch["index"]
            # we use no augmented image for clustering
            data = batch["plain_data"].cuda()
            logit, feature = self.net(data, return_feature=True)
            feature = feature.detach()
            logit = logit.detach()
            score = torch.softmax(logit, dim=1)
            conf, pseudo = torch.max(score, dim=1)
            # evaluation
            for idx in range(len(data)):
                unlabeled_idx_list.append(index[idx].tolist())
                feature_list.append(feature[idx].cpu().tolist())
                unlabeled_conf_list.append(conf[idx].cpu().tolist())
                unlabeled_pseudo_list.append(pseudo[idx].cpu().tolist())
        feature_list = np.array(feature_list)
        unlabeled_idx_list = np.array(unlabeled_idx_list, dtype=int)
        unlabeled_conf_list = np.array(unlabeled_conf_list)
        unlabeled_pseudo_list = np.array(unlabeled_pseudo_list)
        unlabeled_conf_list = sort_array(unlabeled_conf_list, unlabeled_idx_list)
        unlabeled_pseudo_list = sort_array(unlabeled_pseudo_list, unlabeled_idx_list)
        torch.cuda.empty_cache()

        logger.info("Assigning Cluster Labels...")
        cluster_id = self.k_means.cluster(feature_list)
        train_cluster_id = cluster_id[:num_train_data]
        unlabeled_cluster_id = cluster_id[num_train_data:]
        # assign cluster id to samples. Sorted by shuffle-recording index.
        train_cluster_id = sort_array(train_cluster_id, train_idx_list)
        unlabeled_cluster_id = sort_array(unlabeled_cluster_id, unlabeled_idx_list)
        self.labeled_train_loader.dataset.cluster_id = train_cluster_id
        self.unlabeled_train_loader.dataset.cluster_id = unlabeled_cluster_id
        cluster_id = np.concatenate([train_cluster_id, unlabeled_cluster_id])
        # reweighting based on samples in clusters
        cluster_stat = np.zeros(self.num_clusters)
        cluster_id_list, cluster_id_counts = np.unique(cluster_id, return_counts=True)
        for cluster_idx, counts in zip(cluster_id_list, cluster_id_counts):
            cluster_stat[cluster_idx] = counts
        inv_class_freq = 1 / (cluster_stat + 1e-10)
        sample_weight = np.power(inv_class_freq, 0.5)
        sample_weight *= 1 / sample_weight.mean()
        sample_weight_list = np.array([sample_weight[i] for i in cluster_id])
        self.labeled_train_loader.dataset.cluster_reweight = sample_weight_list[
            :num_train_data
        ]
        self.unlabeled_train_loader.dataset.cluster_reweight = sample_weight_list[
            num_train_data:
        ]

        logger.info("In-Distribution Filtering (with OOD Enhancement)...")
        old_train_pseudo_label = self.labeled_train_loader.dataset.pseudo_label
        old_unlabeled_pseudo_label = self.unlabeled_train_loader.dataset.pseudo_label
        old_pseudo_label = np.append(
            old_train_pseudo_label, old_unlabeled_pseudo_label
        ).astype(int)
        new_pseudo_label = (-1 * np.ones_like(old_pseudo_label)).astype(int)
        # process ood confidence for oe loss enhancement (ole)
        new_ood_conf = np.ones_like(old_pseudo_label).astype(float)
        if self.idf_method == "udg":
            total_num_to_filter = 0
            purity_ind_thresh = self.purity_ind_thresh
            purity_ood_thresh = self.purity_ood_thresh
            # pick out clusters with purity over threshold
            for cluster_idx in range(self.num_clusters):
                label_in_cluster, label_counts = np.unique(
                    old_pseudo_label[cluster_id == cluster_idx], return_counts=True
                )
                cluster_size = len(old_pseudo_label[cluster_id == cluster_idx])
                purity = label_counts / cluster_size  # purity list for each label
                # idf
                if np.any(purity > purity_ind_thresh):
                    majority_label = label_in_cluster[purity > purity_ind_thresh][
                        0
                    ]  # first element in the list
                    new_pseudo_label[
                        cluster_id == cluster_idx
                    ] = majority_label  # this might also change some ID but nvm
                    if majority_label > 0:  # ID cluster
                        num_to_filter = len(label_in_cluster == -1)
                        total_num_to_filter += num_to_filter
                # ole
                elif np.any(purity > purity_ood_thresh):
                    majority_label = label_in_cluster[purity > purity_ood_thresh][0]
                    if majority_label == -1:
                        new_ood_conf[cluster_id == cluster_idx] = self.oe_enhance_ratio
            logger.info(f"{total_num_to_filter} number of sample filtered!")

        elif self.idf_method == "conf":
            conf_thresh = self.purity_ind_thresh
            new_pseudo_label[num_train_data:][
                unlabeled_conf_list > conf_thresh
            ] = unlabeled_pseudo_list[unlabeled_conf_list > conf_thresh]
            logger.info(f"Filter {sum(unlabeled_conf_list > conf_thresh)} samples")
        elif self.idf_method == "sort":
            conf_thresh = self.purity_ind_thresh
            num_to_filter = int((1 - conf_thresh) * len(old_unlabeled_pseudo_label))
            new_id_index = np.argsort(-unlabeled_conf_list)[:num_to_filter]
            new_pseudo_label[num_train_data:][new_id_index] = unlabeled_pseudo_list[
                new_id_index
            ]
            logger.info(f"Filter {num_to_filter} samples")
        elif self.idf_method == "none":
            logger.info(f"IDF Disabled, 0 samples filtered")

        self.unlabeled_train_loader.dataset.pseudo_label = new_pseudo_label[
            num_train_data:
        ]
        self.unlabeled_train_loader.dataset.ood_conf = new_ood_conf[num_train_data:]

        logger.info("Randomize Auxiliary Head...")
        if hasattr(self.net, "fc_aux"):
            # reset auxiliary branch
            self.net.fc_aux.weight.data.normal_(mean=0.0, std=0.01)
            self.net.fc_aux.bias.data.zero_()
        else:
            # reset fc for unsupervised learning (baseline)
            self.net.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.net.fc.bias.data.zero_()

        logger.info(
            "Online Clustering Completed! Duration: {:.2f}s".format(
                time.time() - start_time
            ),
        )

    def train_step(self, batch, unlabeled_batch):
        data = batch["data"].cuda()
        unlabeled_data = unlabeled_batch["data"].cuda()

        # concat labeled and unlabeled data
        logits_cls, logits_aux = self.net(data, return_aux=True)
        logits_oe_cls, logits_oe_aux = self.net(unlabeled_data, return_aux=True)

        # classification loss
        concat_logits_cls = torch.cat([logits_cls, logits_oe_cls])
        concat_label = torch.cat(
            [
                batch["label"],
                unlabeled_batch["pseudo_label"].type_as(batch["label"]),
            ]
        )
        loss_cls = F.cross_entropy(
            concat_logits_cls[concat_label != -1],
            concat_label[concat_label != -1].cuda(),
        )
        # oe loss
        concat_softlabel = torch.cat(
            [batch["soft_label"], unlabeled_batch["pseudo_softlabel"]]
        )
        concat_conf = torch.cat([batch["ood_conf"], unlabeled_batch["ood_conf"]])
        loss_oe = rew_sce(
            concat_logits_cls[concat_label == -1],
            concat_softlabel[concat_label == -1].cuda(),
            concat_conf[concat_label == -1].cuda(),
        )
        # aux loss
        concat_logits_aux = torch.cat([logits_aux, logits_oe_aux])
        concat_cluster_id = torch.cat(
            [batch["cluster_id"], unlabeled_batch["cluster_id"]]
        )
        concat_cluster_reweight = torch.cat(
            [batch["cluster_reweight"], unlabeled_batch["cluster_reweight"]]
        )
        loss_aux = rew_ce(
            concat_logits_aux,
            concat_cluster_id.cuda(),
            concat_cluster_reweight.cuda(),
        )

        # loss addition
        loss_oe *= self.lambda_oe
        loss_aux *= self.lambda_aux
        loss = loss_cls + loss_oe + loss_aux

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Record and log metrics
        metrics_dict = {}
        metrics_dict["train/loss_cls"] = loss_cls
        metrics_dict["train/loss_oe"] = loss_oe
        metrics_dict["train/loss_aux"] = loss_aux
        metrics_dict["train/loss"] = loss

        return metrics_dict
