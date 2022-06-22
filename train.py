# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import absolute_import
from __future__ import print_function

import copy
import importlib
import logging
import os
import time
from pathlib import Path

from args import parse_args
from utils.logging import parse_configs_file

args = parse_args()
parse_configs_file(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import data
import models
from utils.logging import (
    parse_prune_stg,
    save_checkpoint,
    create_subdirs,
    clone_results_to_latest_subdir,
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


def main():

    # sanity checks
    if args.exp_mode in ["prune", "finetune"] and not args.resume:
        assert args.source_net, "Provide checkpoint to prune/finetune"

    # create resutls dir (for logs, checkpoints, etc.)
    if args.exp_mode == 'pretrain':
        result_main_dir = os.path.join(Path(args.result_dir), args.exp_mode)
    else:
        result_main_dir = os.path.join(Path(args.result_dir), args.exp_name, args.exp_mode)

    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = os.path.join(
            result_main_dir,
            "{}--k-{:.2f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                n + 1,
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    else:
        os.makedirs(result_main_dir, exist_ok=True)
        result_sub_dir = os.path.join(
            result_main_dir,
            "1--k-{:.2f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    create_subdirs(result_sub_dir)

    parse_prune_stg(args)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
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

    # Create model
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
    prepare_model(model, args)

    # Setup tensorboard writer
    writer = SummaryWriter(os.path.join(result_sub_dir, "tensorboard"))

    # Dataloader
    D = data.__dict__[args.dataset](args, normalize=args.normalize)
    train_loader, test_loader = D.data_loaders()

    logger.info(
        f"Dataset: {args.dataset}, D: {D}, num_train: {len(train_loader.dataset)}, num_test:{len(test_loader.dataset)}")

    # Semi-sup dataloader
    if args.is_semisup:
        logger.info("Using semi-supervised training")
        sm_loader = get_semisup_dataloader(args, D.tr_train)
    else:
        sm_loader = None

    # autograd
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
    logger.info([criterion, optimizer, lr_policy])

    # train & val method
    trainer = importlib.import_module(f"trainer.{args.trainer}").train
    val = getattr(importlib.import_module("utils.eval"), args.val_method)

    # Load source_net (if checkpoint provided). Only load the state_dict (required for pruning and fine-tuning)
    if args.source_net:
        if args.exp_mode == 'prune':
            hydra_dir = os.getcwd()
            args.source_net = '/'.join(hydra_dir.split('/')[:-1] + [args.source_net])
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
            if args.exp_mode == 'prune':
                checkpoint_dict = {k.replace("module.basic_model.", ""): v for k, v in checkpoint_dict.items() if k.find('popup_scores') == -1}
                model_dict.update(checkpoint_dict)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(checkpoint_dict)
            logger.info("=> loaded checkpoint '{}'".format(args.source_net))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.source_net))

    # Init scores once source net is loaded.
    # NOTE: scaled_init_scores will overwrite the scores in the pre-trained net.
    if args.scaled_score_init:
        initialize_scaled_score(model, args.prune_reg)

    # Scaled random initialization. Useful when training a high sparse net from scratch.
    # If not used, a sparse net (without batch-norm) from scratch will not coverge.
    # With batch-norm its not really necessary.
    if args.scale_rand_init:
        scale_rand_init(model, args.k)

    # Scaled random initialization. Useful when training a high sparse net from scratch.
    # If not used, a sparse net (without batch-norm) from scratch will not coverge.
    # With batch-norm its not really necessary.
    if args.scale_rand_init:
        scale_rand_init(model, args.k)

    if args.snip_init:
        snip_init(model, criterion, optimizer, train_loader, device, args)

    assert not (args.source_net and args.resume), (
        "Incorrect setup: "
        "resume => required to resume a previous experiment (loads all parameters)|| "
        "source_net => required to start pruning/fine-tuning from a source model (only load state_dict)"
    )
    # resume (if checkpoint provided). Continue training with preiovus settings.
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Evaluate
    if args.evaluate or args.exp_mode in ["prune", "finetune"]:
        p1_bn, _, p1, _, loss, adv_loss = val(model, device, test_loader, criterion, args, writer)
        logger.info(
            f"Benign validation accuracy {args.val_method} for source-net: {p1_bn}, Adversarial validation accuracy {args.val_method} for source-net: {p1}")
        if args.evaluate:
            return

    best_prec1 = 0

    show_gradients(model)

    if args.source_net and args.exp_mode != 'pretrain':
        last_ckpt = checkpoint["state_dict"]
    else:
        last_ckpt = copy.deepcopy(model.state_dict())

    # Capture Loss, Adv Loss, Benign Acc & Adv Acc
    losses = []
    adv_losses = []
    acc_ben = []
    acc_adv = []
    # Start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + args.warmup_epochs):
        lr_policy(epoch)  # adjust learning rate

        # train
        trainer(
            model,
            device,
            train_loader,
            sm_loader,
            criterion,
            optimizer,
            epoch,
            args,
            writer,
        )
        # evaluate on test set
        if args.val_method == "smooth":
            prec1, radii = val(
                model, device, test_loader, criterion, args, writer, epoch
            )
            logger.info(f"Epoch {epoch}, mean provable Radii  {radii}")
        if args.val_method == "mixtrain" and epoch <= args.schedule_length:
            prec1 = 0.0
        else:
            prec1_benign, _, prec1, _, loss, adv_loss = val(model, device, test_loader, criterion, args, writer, epoch)
            losses.append(loss)
            adv_losses.append(adv_loss)
            acc_ben.append(prec1_benign.item())
            acc_adv.append(prec1.item())

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args,
            result_dir=os.path.join(result_sub_dir, "checkpoint"),
            save_dense=args.save_dense,
        )

        logger.info(
            f"Epoch {epoch}, val-method {args.val_method}, benign validation accuracy {prec1_benign}, adversarial validation accuracy {prec1}, best_prec {best_prec1}"
        )
        if args.exp_mode in ["prune", "finetune"]:
            logger.info(
                "Pruned model: {:.2f}%".format(
                    current_model_pruned_fraction(
                        model, os.path.join(result_sub_dir, "checkpoint"), verbose=False
                    )
                )
            )
        # clone results to latest subdir (sync after every epoch)
        # Latest_subdir: stores results from latest run of an experiment.
        clone_results_to_latest_subdir(
            result_sub_dir, os.path.join(result_main_dir, "latest_exp")
        )

        # Check what parameters got updated in the current epoch.
        sw, ss = sanity_check_paramter_updates(model, last_ckpt)
        logger.info(
            f"Sanity check (exp-mode: {args.exp_mode}): Weight update - {sw}, Scores update - {ss}"
        )

        print(f"Time since start of training: {float(time.time() - start_time) / 60} minutes")

    end_time = time.time()
    print(
        f"Total training time: {end_time - start_time} seconds. These are {float((end_time - start_time) / 3600)} "
        f"hours")


if __name__ == "__main__":
    main()
