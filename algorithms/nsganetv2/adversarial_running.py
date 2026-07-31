import copy

import torchvision
from ofa.imagenet_classification.run_manager import RunManager
import os
import random
import time
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from tqdm import tqdm
from torch.utils.flop_counter import FlopCounterMode
from ofa.utils import (
    get_net_info,
    cross_entropy_loss_with_soft_target,
    cross_entropy_with_label_smoothing,
)
from ofa.utils import (
    AverageMeter,
    accuracy,
    write_log,
    mix_images,
    mix_labels,
    init_models,
)
from ofa.utils import MyRandomResizedCrop

__all__ = ["RunManager"]

from adversarial import fgsm_simple


class AdvRunManager(RunManager):

    def __init__(self, path, net, run_config, init=True, measure_latency=None, no_gpu=False):
        super().__init__(path, net, run_config, init, measure_latency, no_gpu)

    def validate_adv(
        self,
        epoch=0,
        is_test=False,
        run_str="",
        net=None,
        data_loader=None,
        no_logs=False,
        train_mode=False
    ):
        if net is None:
            net = self.net
        if not isinstance(net, nn.DataParallel):
            net = nn.DataParallel(net)

        if data_loader is None:
            data_loader = (
                self.run_config.test_loader if is_test else self.run_config.valid_loader
            )

        if train_mode:
            net.train()
        else:
            net.eval()

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()

        total = 0
        std_loss_mean = 0
        adv_loss_mean = 0
        total_loss_mean = 0
        start_time = time.time()

        with tqdm(
            total=len(data_loader),
            desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
            disable=no_logs,
        ) as t:
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                images.requires_grad = True
                adv_images, std_logits = fgsm_simple(net, images, labels)
                std_loss = self.test_criterion(std_logits, labels)
                adv_logits = net(adv_images)
                adv_loss = self.test_criterion(adv_logits, labels)
                loss = std_loss * 0.5 + adv_loss * 0.5

                # measure accuracy and record loss
                self.update_metric(metric_dict, std_logits, labels)

                losses.update(loss.item(), images.size(0))
                t.set_postfix(
                    {
                        "loss": losses.avg,
                        **self.get_metric_vals(metric_dict, return_dict=True),
                        "img_size": images.size(2),
                    }
                )
                t.update(1)
                total += images.size(0)
                std_loss_mean += std_loss.item()
                adv_loss_mean += adv_loss.item()
                total_loss_mean += loss.item()
            std_loss_mean /= total
            adv_loss_mean /= total
            total_loss_mean /= total

        batch = next(iter(data_loader))
        model = net.module if isinstance(net, torch.nn.DataParallel) else net

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        #model = copy.deepcopy(model).to(device)
        params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params = round(float(params_num) / 1e6, 4)
        x = torch.randn(1, 3, 32, 32).to(device)
        with FlopCounterMode(display=False) as flop_counter:
            model(x)
        flops = round(float(flop_counter.get_total_flops()) / 1e6, 4)

        return total_loss_mean, std_loss_mean, adv_loss_mean, flops, params, self.get_metric_vals(metric_dict)

    def validate_all_resolution_adv(self, epoch=0, is_test=False, net=None):
        if net is None:
            net = self.network
        if isinstance(self.run_config.data_provider.image_size, list):
            img_size_list, loss_list, top1_list, top5_list = [], [], [], []
            for img_size in self.run_config.data_provider.image_size:
                img_size_list.append(img_size)
                self.run_config.data_provider.assign_active_img_size(img_size)
                self.reset_running_statistics(net=net)
                total_loss_mean, std_loss_mean, adv_loss_mean, flops, params, (top1, top5) = self.validate_adv(epoch, is_test, net=net)
                loss_list.append(total_loss_mean)
                top1_list.append(top1)
                top5_list.append(top5)
            return img_size_list, loss_list, top1_list, top5_list
        else:
            total_loss_mean, std_loss_mean, adv_loss_mean, flops, params, (top1, top5) = self.validate_adv(epoch, is_test, net=net)
            return (
                [self.run_config.data_provider.active_img_size],
                [total_loss_mean],
                [top1],
                [top5],
            )

    def train_one_epoch_adv(self, args, epoch, warmup_epochs=0, warmup_lr=0):
        # switch to train mode
        self.net.train()
        MyRandomResizedCrop.EPOCH = epoch  # required by elastic resolution

        nBatch = len(self.run_config.train_loader)

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()
        data_time = AverageMeter()
        with tqdm(
            total=nBatch,
            desc="{} Train Epoch #{}".format(self.run_config.dataset, epoch + 1),
        ) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                MyRandomResizedCrop.BATCH = i
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        self.optimizer,
                        warmup_epochs * nBatch,
                        nBatch,
                        epoch,
                        i,
                        warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(
                        self.optimizer, epoch - warmup_epochs, i, nBatch
                    )

                images, labels = images.to(self.device), labels.to(self.device)
                target = labels
                if isinstance(self.run_config.mixup_alpha, float):
                    # transform data
                    lam = random.betavariate(
                        self.run_config.mixup_alpha, self.run_config.mixup_alpha
                    )
                    images = mix_images(images, lam)
                    labels = mix_labels(
                        labels,
                        lam,
                        self.run_config.data_provider.n_classes,
                        self.run_config.label_smoothing,
                    )

                # soft target
                if args.teacher_model is not None:
                    args.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = args.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)
                images.requires_grad = True
                adv_images, std_logits = fgsm_simple(self.net, images, target)
                std_loss = self.train_criterion(std_logits, labels)
                # compute output
                adv_output = self.net(adv_images)
                adv_loss = self.train_criterion(adv_output, labels)
                loss = std_loss * 0.5 + adv_loss * 0.5

                if args.teacher_model is None:
                    loss_type = "ce"
                else:
                    if args.kd_type == "ce":
                        kd_loss = cross_entropy_loss_with_soft_target(
                            std_logits, soft_label
                        )
                    else:
                        kd_loss = F.mse_loss(std_logits, soft_logits)
                    loss = args.kd_ratio * kd_loss + loss
                    loss_type = "%.1fkd+ce" % args.kd_ratio

                # compute gradient and do SGD step
                self.net.zero_grad()  # or self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                self.update_metric(metric_dict, std_logits, target)

                t.set_postfix(
                    {
                        "loss": losses.avg,
                        **self.get_metric_vals(metric_dict, return_dict=True),
                        "img_size": images.size(2),
                        "lr": new_lr,
                        "loss_type": loss_type,
                        "data_time": data_time.avg,
                    }
                )
                t.update(1)
                end = time.time()
        return losses.avg, self.get_metric_vals(metric_dict)

    def train_adv(self, args, warmup_epoch=0, warmup_lr=0):
        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epoch):
            train_loss, (train_top1, train_top5) = self.train_one_epoch_adv(
                args, epoch, warmup_epoch, warmup_lr
            )
            """
            if (epoch + 1) % self.run_config.validation_frequency == 0:
                img_size, val_loss, val_acc, val_acc5 = self.validate_all_resolution_adv(
                    epoch=epoch, is_test=False
                )

                is_best = np.mean(val_acc) > self.best_acc
                self.best_acc = max(self.best_acc, np.mean(val_acc))
                val_log = "Valid [{0}/{1}]\tloss {2:.3f}\t{5} {3:.3f} ({4:.3f})".format(
                    epoch + 1 - warmup_epoch,
                    self.run_config.n_epochs,
                    np.mean(val_loss),
                    np.mean(val_acc),
                    self.best_acc,
                    self.get_metric_names()[0],
                )
                val_log += "\t{2} {0:.3f}\tTrain {1} {top1:.3f}\tloss {train_loss:.3f}\t".format(
                    np.mean(val_acc5),
                    *self.get_metric_names(),
                    top1=train_top1,
                    train_loss=train_loss
                )
                for i_s, v_a in zip(img_size, val_acc):
                    val_log += "(%d, %.3f), " % (i_s, v_a)
                self.write_log(val_log, prefix="valid", should_print=False)
            else:
                is_best = False
            """
            self.save_model(
                {
                    "epoch": epoch,
                    "best_acc": self.best_acc,
                    "optimizer": self.optimizer.state_dict(),
                    "state_dict": self.network.state_dict(),
                },
                #is_best=is_best,
            )

    def reset_running_statistics(
        self, net=None, subset_size=2000, subset_batch_size=200, data_loader=None
    ):
        from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

        if net is None:
            net = self.network
        if data_loader is None:
            data_loader = self.run_config.random_sub_train_loader(
                subset_size, subset_batch_size
            )
        set_running_statistics(net, data_loader)