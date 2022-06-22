# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import absolute_import
from __future__ import print_function

import copy
import csv
import importlib
import logging
import os
import time
from itertools import zip_longest
from pathlib import Path

from args import parse_args
from utils.logging import parse_configs_file
from sklearn.metrics import balanced_accuracy_score

args = parse_args()
parse_configs_file(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch
import torch.nn as nn

import data as Data
import models
from utils.logging import (
    parse_prune_stg,
    AverageMeter,
    ProgressMeter,
    create_subdirs,
    clone_results_to_latest_subdir,
)
from utils.adv import (
    pgd_whitebox,
    fgsm_whitebox,
    cw_whitebox
)
from utils.model import (
    get_layers,
    prepare_model,
    initialize_scaled_score,
    scale_rand_init,
    show_gradients,
    current_model_pruned_fraction,
    sanity_check_paramter_updates,
    snip_init,
)
from utils.schedules import get_lr_policy, get_optimizer
from utils.semisup import get_semisup_dataloader


# TODO: update wrn, resnet models. Save both subnet and dense version.
# TODO: take care of BN, bias in pruning, support structured pruning

ATTACK_LIST = ['pgd']  # ['fgsm', 'pgd', 'cw']


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():

    # sanity checks
    if args.exp_mode in ["prune", "finetune"] and not args.resume:
        assert args.source_net, "Provide checkpoint to prune/finetune"

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(Path(args.result_dir), args.stg_mode, args.exp_name, args.exp_mode)

    # if os.path.exists(result_main_dir):
    #     n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
    #     result_sub_dir = os.path.join(
    #         result_main_dir,
    #         "{}--k-{:.2f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
    #             n + 1,
    #             args.k,
    #             args.trainer,
    #             args.lr,
    #             args.epochs,
    #             args.warmup_lr,
    #             args.warmup_epochs,
    #         ),
    #     )
    # else:
    #     os.makedirs(result_main_dir, exist_ok=True)
    #     result_sub_dir = os.path.join(
    #         result_main_dir,
    #         "1--k-{:.2f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
    #             args.k,
    #             args.trainer,
    #             args.lr,
    #             args.epochs,
    #             args.warmup_lr,
    #             args.warmup_epochs,
    #         ),
    #     )
    # create_subdirs(result_sub_dir)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    # logger.addHandler(
    #     logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    # )
    logger.info(args)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Select GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    num_gpus = len(args.gpu.strip().split(","))
    # gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    gpu_list = [i for i in range(num_gpus)]
    device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")

    # Create identical models for different tasks
    cl, ll = get_layers(args.layer_type)
    if len(gpu_list) > 1:
        print("Using multiple GPUs")
        model = nn.DataParallel(
            models.__dict__[args.arch](
                cl, ll, args.init_type, num_classes=args.num_classes, prune_reg=args.prune_reg, task_mode=args.exp_mode
            ),
            gpu_list,
        ).to(device)
    else:
        model = models.__dict__[args.arch](
            cl, ll, args.init_type, num_classes=args.num_classes, prune_reg=args.prune_reg, task_mode=args.exp_mode
        ).to(device)

    logger.info(model)

    # Customize models for training/pruning/fine-tuning
    log_outdir = os.path.join(args.result_dir, args.stg_mode, args.exp_name)
    create_subdirs(log_outdir)

    parse_prune_stg(args)
    prepare_model(model, args)

    # Dataloader
    D = Data.__dict__[args.dataset](args, normalize=args.normalize)
    train_loader, test_loader = D.data_loaders()

    logger.info(
        f"Dataset: {args.dataset}, D: {D}, num_train: {len(train_loader.dataset)}, num_test:{len(test_loader.dataset)}")

    # autograd
    criterion = nn.CrossEntropyLoss()
    logger.info([criterion])

    # Load source_net (if checkpoint provided). Only load the state_dict (required for pruning and fine-tuning)
    if args.source_net:
        if os.path.isfile(args.source_net):
            logger.info("=> loading source model from '{}'".format(args.source_net))
            checkpoint = torch.load(args.source_net, map_location=device)
            model_dict = model.state_dict()
            """
            if (args.arch == "resnet18" or args.arch == "ResNet18") and args.exp_mode == 'prune':
                checkpoint_dict = checkpoint['net']
            else:
                checkpoint_dict = checkpoint['state_dict']
            """
            checkpoint_dict = checkpoint['state_dict']
            model.load_state_dict(checkpoint_dict)
            logger.info("=> loaded checkpoint '{}'".format(args.source_net))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    assert args.source_net, (
        "Incorrect setup: "
        "source_net => required to start pruning/fine-tuning from a source model (only load state_dict)"
    )

    # Run attack
    for attack in ATTACK_LIST:
        args.attack_eval = attack

        """
            Evaluate on adversarial validation set inputs.
        """

        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        adv_losses = AverageMeter("Adv_Loss", ":.4f")
        top1 = AverageMeter("Acc_1", ":6.2f")
        top5 = AverageMeter("Acc_5", ":6.2f")
        adv_top1 = AverageMeter("Adv-Acc_1", ":6.2f")
        adv_top5 = AverageMeter("Adv-Acc_5", ":6.2f")
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, losses, adv_losses, top1, top5, adv_top1, adv_top5],
            prefix="Test: ",
        )

        if args.dataset == 'imagenet':
            mean = torch.Tensor(np.array(args.mean)[:, np.newaxis, np.newaxis])
            std = torch.Tensor(np.array(args.std)[:, np.newaxis, np.newaxis])
        else:
            mean = torch.Tensor(np.array([0.0, 0.0, 0.0])[:, np.newaxis, np.newaxis])
            std = torch.Tensor(np.array([1.0, 1.0, 1.0])[:, np.newaxis, np.newaxis])

        mean = mean.expand(3, args.image_dim, args.image_dim).cuda()
        std = std.expand(3, args.image_dim, args.image_dim).cuda()

        # switch to evaluate mode
        model.eval()

        nat_labels = []
        nat_preds_all = []
        adv_preds_all = []

        with torch.no_grad():

            for i, data in enumerate(test_loader):

                images, target = data[0].to(device), data[1].to(device)

                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # compute output
                images = images - mean
                images.div_(std)

                # Compute nat output
                output_nat = model(images)

                # adversarial images
                if args.attack_eval == 'pgd':
                    attacker = pgd_whitebox
                elif args.attack_eval == 'fgsm':
                    attacker = fgsm_whitebox
                elif args.attack_eval == 'cw':
                    attacker = cw_whitebox
                    args.num_steps = 50
                else:
                    raise NameError(f'{args.attack_eval} is not supported for white-box attack!')

                images = attacker(
                    model,
                    images,
                    target,
                    device,
                    args.epsilon,
                    args.num_steps,
                    args.step_size,
                    args.clip_min,
                    args.clip_max,
                    is_random=not args.const_init,
                )

                # compute output
                output_adv = model(images)

                # measure accuracy and record loss
                _, nat_preds = output_nat.topk(1, 1, True, True)
                nat_preds = nat_preds.view(-1).cpu().numpy()
                nat_labels = np.append(nat_labels, target.cpu().numpy().squeeze())
                nat_preds_all = np.append(nat_preds_all, nat_preds)

                _, adv_preds = output_adv.topk(1, 1, True, True)
                adv_preds = adv_preds.view(-1).cpu().numpy()
                adv_preds_all = np.append(adv_preds_all, adv_preds)

                nat_acc1, nat_acc5 = accuracy(output_nat, target, topk=(1, 5))
                adv_acc1, adv_acc5 = accuracy(output_adv, target, topk=(1, 5))

                top1.update(nat_acc1[0], images.size(0))
                top5.update(nat_acc5[0], images.size(0))
                adv_top1.update(adv_acc1[0], images.size(0))
                adv_top5.update(adv_acc5[0], images.size(0))

                if (i + 1) % args.print_freq == 0:
                    progress.display(i)

            progress.display(i)  # print final results

        # p1_bn, _, p1, _, loss, adv_loss = val(model, device, test_loader, criterion, args, writer)
        top1_bacc = balanced_accuracy_score(nat_labels, nat_preds_all) * 100.0
        adv_top1_bacc = balanced_accuracy_score(nat_labels, adv_preds_all) * 100.0
        logger.info(
            f"BALANCED ACC: Benign validation accuracy {args.val_method} for source-net: {top1_bacc}, Adversarial {attack.upper()} validation accuracy for source-net: {adv_top1_bacc}")
        logger.info(
            f"STANDARD ACC: Benign validation accuracy {args.val_method} for source-net: {top1.avg}, Adversarial {attack.upper()} validation accuracy for source-net: {adv_top1.avg}")


if __name__ == "__main__":
    main()
