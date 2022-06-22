import torch
import torch.nn as nn
import torchvision

import os
import math
import numpy as np
from copy import copy

from models.layers import SubnetConv, SubnetLinear

# TODO: avoid freezing bn_params
# Some utils are borrowed from https://github.com/allenai/hidden-networks
def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True


def set_prune_rate_model(model, args):
    for block_n, v in model.named_modules():

        if hasattr(v, "set_prune_rate"):

            if block_n.find('sub_block') != -1:
                # Actually sub_block doesn't exit in forward(), here is just like placeholder for no-bug
                if args.exp_mode == 'pretrain':
                    v.set_prune_rate(args.k, 1.0)
                else:
                    v.set_prune_rate(args.k, args.k)
            else:
                if args.exp_mode == 'pretrain':
                    v.set_prune_rate(args.k, 1.0)
                else:
                    v_name = v._get_name()
                    if block_n.lower().find('shortcut') != -1 or block_n.lower().find('downsample') != -1:
                        k = args.shortcut_k.pop(0)
                    elif v_name == 'SubnetConv':
                        k = args.conv_k.pop(0)
                    elif v_name == 'SubnetLinear':
                        k = args.fc_k.pop(0)
                    else:
                        raise NameError('{} has no pruning rate!'.format(v_name))

                    print('$$$ {} prune rate {:.8f} on {}'.format(args.prune_reg, k, v))
                    v.set_prune_rate(args.k, k)


def get_layers(layer_type):
    """
        Returns: (conv_layer, linear_layer)
    """
    if layer_type == "dense":
        return nn.Conv2d, nn.Linear
    elif layer_type == "subnet":
        return SubnetConv, SubnetLinear
    else:
        raise ValueError("Incorrect layer type")


def show_gradients(model):
    for i, v in model.named_parameters():
        print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")


def snip_init(model, criterion, optimizer, train_loader, device, args):
    print("Using SNIP initialization")
    assert args.exp_mode == "pretrain"
    optimizer.zero_grad()
    # init the score with kaiming normal init
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            nn.init.kaiming_normal_(m.popup_scores, mode="fan_in")

    set_prune_rate_model(model, args)
    unfreeze_vars(model, "popup_scores")

    # take a forward pass and get gradients
    for _, data in enumerate(train_loader):
        images, target = data[0].to(device), data[1].to(device)

        output = model(images)
        loss = criterion(output, target)

        loss.backward()
        break

    # update scores with their respective connection sensitivty
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            print(m.popup_scores.data)
            m.popup_scores.data = m.popup_scores.grad.data.abs()
            print(m.popup_scores.data)

    # update k back to args.k.
    # set_prune_rate_model(model, args.k)
    set_prune_rate_model(model, args)
    freeze_vars(model, "popup_scores")


def initialize_scores(model, init_type):
    print(f"Initialization relevance score with {init_type} initialization")
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.popup_scores)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.popup_scores)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )


def initialize_scaled_score(model, prune_reg='weight'):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES) | Prune_Reg: {}".format(prune_reg)
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")

            if prune_reg == 'weight':
                # Weight Pruning
                # """
                # Close to kaiming unifrom init
                m.popup_scores.data = (
                    math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
                )
                # """

            elif prune_reg == 'channel':
                # Channel Prune
                # """
                reshaped_weights = torch.sum(torch.abs(m.weight.data.reshape(m.weight.data.shape[1],-1)),dim=1)
                if type(m) == SubnetConv:
                    channel_popup_scores = (math.sqrt(6/n)*reshaped_weights / torch.max(torch.abs(reshaped_weights))).reshape(1,m.weight.data.shape[1],1,1)
                else:
                    channel_popup_scores = (
                            math.sqrt(6 / n) * reshaped_weights / torch.max(torch.abs(reshaped_weights))).reshape(
                        1, m.weight.data.shape[1])
                m.popup_scores.data = channel_popup_scores
                # """
            else:
                raise NameError('Please check prune_reg, current "{}" is not in [weight, channel] !'.format(prune_reg))



def scale_rand_init(model, k):
    print(
        f"Initializating random weight with scaling by 1/sqrt({k}) | Only applied to CONV & FC layers"
    )
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # print(f"previous std = {torch.std(m.weight.data)}")
            m.weight.data = 1 / math.sqrt(k) * m.weight.data
            # print(f"new std = {torch.std(m.weight.data)}")


def initialize_lamp_rate(model, k_list):
    print(f'Initializing layer-wise rate with LAMP strategy')

    all_k = copy(k_list)
    for m in model.modules():
        if hasattr(m, "k_rate"):
            m.k_rate.data = all_k.pop(0)


def prepare_model(model, args):
    """
        1. Set model pruning rate
        2. Set gradients base of training mode.
    """

    set_prune_rate_model(model, args)

    if args.exp_mode == "pretrain":
        print(f"#################### Pre-training network ####################")
        print(f"===>>  gradient for importance_scores: None  | training weights only")
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")

    elif args.exp_mode == "prune":
        print(f"#################### Pruning network ####################")
        print(f"===>>  gradient for weights: None  | training importance scores only")

        unfreeze_vars(model, "popup_scores")
        freeze_vars(model, "weight", args.freeze_bn)
        freeze_vars(model, "bias", args.freeze_bn)

    elif args.exp_mode == "finetune":
        print(f"#################### Fine-tuning network ####################")
        print(
            f"===>>  gradient for importance_scores: None  | fine-tuning important weigths only"
        )
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
    else:
        assert False, f"{args.exp_mode} mode is not supported"

    initialize_scores(model, args.scores_init_type)


def subnet_to_dense(subnet_dict, p):
    """
        Convert a subnet state dict (with subnet layers) to dense i.e., which can be directly 
        loaded in network with dense layers.
    """
    dense = {}

    # load dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" not in k:
            dense[k] = v

    # update dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" in k:
            s = torch.abs(subnet_dict[k])

            out = s.clone()
            _, idx = s.flatten().sort()
            j = int((1 - p) * s.numel())

            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1
            dense[k.replace("popup_scores", "weight")] = (
                subnet_dict[k.replace("popup_scores", "weight")] * out
            )
    return dense


def dense_to_subnet(model, state_dict):
    """
        Load a dict with dense-layer in a model trained with subnet layers. 
    """
    model.load_state_dict(state_dict, strict=False)


def current_model_pruned_fraction(model, result_dir, verbose=True):
    """
        Find pruning raio per layer. Return average of them.
        Result_dict should correspond to the checkpoint of model.
    """

    # load the dense models
    path = os.path.join(result_dir, "checkpoint_dense.pth.tar")

    pl = []
    zero_weights = 0
    total_params = 0

    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for i, v in model.named_modules():
            if isinstance(v, (nn.Conv2d, nn.Linear)):
                if i + ".weight" in state_dict.keys():
                    d = state_dict[i + ".weight"].data.cpu().numpy()
                    """
                    p = 100 * np.sum(d == 0) / np.size(d)
                    pl.append(p)
                    """
                    zero_weights += np.sum(d==0)
                    total_params += np.size(d)
                    # if verbose:
                    #     print(i, v, p)
        # return np.mean(pl)
        return (float(zero_weights)/float(total_params))*100.
        

def sanity_check_paramter_updates(model, last_ckpt):
    """
        Check whether weigths/popup_scores gets updated or not compared to last ckpt.
        ONLY does it for 1 layer (to avoid computational overhead)
    """
    for i, v in model.named_modules():
        if hasattr(v, "weight") and hasattr(v, "popup_scores"):
            if getattr(v, "weight") is not None:
                w1 = getattr(v, "weight").data.cpu()
                w2 = last_ckpt[i + ".weight"].data.cpu()
            if getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                s2 = last_ckpt[i + ".popup_scores"].data.cpu()
            return not torch.allclose(w1, w2), not torch.allclose(s1, s2)