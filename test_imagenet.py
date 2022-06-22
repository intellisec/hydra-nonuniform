# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
import argparse
import importlib
import time
import logging
from pathlib import Path
import copy
import math

from args import parse_args
from utils.logging import parse_configs_file

args = parse_args()

# args.configs = './configs/config-ResNet50-imagenet.yml'
parse_configs_file(args)

# args.batch_size = 64
# args.gpu = '1,2'
# args.exp_mode = 'finetune'
# args.k = 0.5
# args.prune_reg = 'channel'
# args.stg_id = '05_0'
# args.num_steps = 20
# args.resume = False
# args.source_net = f'./trained_models/resnet50/imagenet_{args.prune_reg}_{args.stg_id}/finetune/latest_exp/checkpoint/checkpoint.pth.tar'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn

import models
import data
from utils.schedules import get_lr_policy, get_optimizer
from utils.logging import (
    parse_prune_stg,
    save_checkpoint,
    create_subdirs,
    clone_results_to_latest_subdir,
)
from utils.semisup import get_semisup_dataloader
from utils.model import (
    get_layers,
    prepare_model,
    initialize_scaled_score,
    show_gradients,
    current_model_pruned_fraction,
    sanity_check_paramter_updates,
)


# TODO: update wrn, resnet models. Save both subnet and dense version.
# TODO: take care of BN, bias in pruning, support structured pruning


def main():

    # sanity checks
    if args.exp_mode in ["prune", "finetune"] and not args.resume:
        assert args.source_net, "Provide checkpoint to prune/finetune"

    # create resutls dir (for logs, checkpoints, etc.)
    exp_name = '_'.join([args.dataset.lower(), args.prune_reg, args.stg_id])
    result_main_dir = os.path.join(Path(args.result_dir), args.stg_mode, exp_name, args.exp_mode)

    parse_prune_stg(args)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_main_dir, "eval_attacks.log"), "a")
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

    # Dataloader
    D = data.__dict__[args.dataset](args, normalize=args.normalize)
    train_loader, test_loader = D.data_loaders()

    logger.info(
        f"Dataset: {args.dataset}, D: {D}, num_train: {len(train_loader.dataset)}, num_test:{len(test_loader.dataset)}")

    # autograd
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
    logger.info([criterion, optimizer, lr_policy])

    # eval methods
    val = getattr(importlib.import_module("utils.eval"), 'adv_imagenet')
    bng_val = getattr(importlib.import_module("utils.eval"), 'imagenet_base')

    # Load source_net (if checkpoint provided). Only load the state_dict (required for pruning and fine-tuning)
    if args.source_net:
        if os.path.isfile(args.source_net):
            logger.info("=> loading source model from '{}'".format(args.source_net))
            checkpoint = torch.load(args.source_net, map_location=device)
            # model.load_state_dict(
            #     checkpoint["state_dict"], strict=False
            # )  # allows loading dense models
            model_dict = model.state_dict()
            """
            if (args.arch == "resnet18" or args.arch == "ResNet18") and args.exp_mode == 'prune':
                checkpoint_dict = checkpoint['net']
            else:
                checkpoint_dict = checkpoint['state_dict']
            """
            checkpoint_dict = checkpoint['state_dict']
            if args.exp_mode == 'prune' or args.evaluate:
                checkpoint_dict = {k.replace("module.basic_model.", ""): v for k, v in checkpoint_dict.items() if k.find('popup_scores') == -1}
                model_dict.update(checkpoint_dict)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(checkpoint_dict)
            logger.info("=> loaded checkpoint '{}'".format(args.source_net))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Init scores once source net is loaded.
    # NOTE: scaled_init_scores will overwrite the scores in the pre-trained net.
    if args.scaled_score_init:
        initialize_scaled_score(model, args.prune_reg)

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
        bng_p1, bng_p5 = bng_val(model, device, test_loader, criterion, args, writer=None)
        logger.info(
            f"Benign validation accuracy for source-net: Top-1 {bng_p1}, Top-5 {bng_p5}")

        for adv in ['fgsm', 'pgd', 'cw']:
            if adv == 'fgsm':
                args.num_steps = 1
                args.step_size = 0.0156
            else:
                args.num_steps = 20
                args.step_size = 0.00392
            p1, p5 = val(model, device, test_loader, args, writer=None, attack=adv)
            logger.info(
                f"Adversarial validation accuracy by {adv.upper()}: Top-1 {p1}, Top-5 {p5}")


if __name__ == "__main__":
    main()
