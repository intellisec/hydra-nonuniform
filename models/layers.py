import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

conv_nr = 0
linear_nr = 0

# https://github.com/allenai/hidden-networks
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, prune_reg='weight'):
        # Get the subnetwork by sorting the scores and using the top k%

        if prune_reg == 'weight':
            # """ Weight pruning
            out = scores.clone()
            _, idx = scores.flatten().sort()
            j = int((1 - k) * scores.numel())

            # flat_out and out access the same memory.
            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1
            # """

        elif prune_reg == 'channel':
            """ Channel pruning without changed score mask (only for filter prune)
            out = scores.clone()
            # print(scores.shape)
            kept_weights = torch.topk(torch.linalg.norm(out.reshape(out.shape[0], -1), 1, dim=1),
                                      int(k * out.shape[0])).indices
            out[:] = 0
            out[kept_weights] = 1
            """
            # """ Channel pruning with changed score mask
            out = scores.clone()
            kept_weights = torch.topk(out, k=int(round(k*out.shape[1])), dim=1).indices
            out = torch.transpose(out, 0,1)
            out[:] = 0
            out[kept_weights] = 1
            out = torch.transpose(out, 0,1)
            # """

        else:
            raise NameError('Please check prune_reg, current "{}" is not in [weight, channel] !'.format(prune_reg))

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None


class SubnetConv(nn.Conv2d):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off by default.

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            prune_reg='weight',
            task_mode='prune'
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.prune_reg = prune_reg

        if self.prune_reg == 'weight':
            # Weight pruning or Filter Pruning
            self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        elif self.prune_reg == 'channel' and task_mode == 'finetune':
            # Channel Finetuning or Resume Pruning
            self.popup_scores = Parameter(torch.Tensor(torch.Size([1,self.weight.shape[1],1,1])))
        elif self.prune_reg == 'channel' and task_mode == 'prune':
            # Channel Pruning
            self.popup_scores = Parameter(torch.Tensor(torch.Size([self.weight.shape[0], 1,1,1])))
        else:
            raise NameError('prune_reg "{}" or task_mode "{}" are not correct!'.format(prune_reg, task_mode))

        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0

    def set_prune_rate(self, global_k, k):
        self.k = k
        self.prune_k = k

    def forward(self, x):
        """ Unstructured comparison
        remaining_weights = int(self.k * len(self.weight.flatten()))
        idx_same_top_weights_scores = list(
            set(torch.topk(self.weight.abs().flatten(), remaining_weights).indices.tolist()).intersection(
                set(torch.topk(self.popup_scores.abs().flatten(), remaining_weights).indices.tolist())))
        num_remaining_weights = len(idx_same_top_weights_scores)
        print(
            f"SubnetConv: Number of same indices for scores and weights that are left after pruning: "
            f"{num_remaining_weights}. These are {float(num_remaining_weights / remaining_weights)} percent of the "
            f"weights kept.")
        """
        """ Structured Comparison
        remaining_filters = int(self.k * self.weight.shape[0])
        idx_same_top_weights_scores = list(set(
            torch.topk(torch.linalg.norm(self.weight.abs().reshape(self.weight.shape[0], -1), 1, dim=1),
                       remaining_filters).indices.tolist()).intersection(
            torch.topk(torch.linalg.norm(self.popup_scores.abs().reshape(self.popup_scores.shape[0], -1), 1, dim=1),
                       remaining_filters).indices.tolist()))
        num_remaining_filters = len(idx_same_top_weights_scores)
        print(
            f"SubnetConv: Number of same indices for filters that are left after pruning using scores or weights : "
            f"{num_remaining_filters}. These are {float(num_remaining_filters / remaining_filters)} percent of the "
            f"filters kept.")
        """

        adj = GetSubnet.apply(self.popup_scores.abs(), self.prune_k, self.prune_reg)

        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetLinear(nn.Linear):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.

    def __init__(self, in_features, out_features, bias=True, prune_reg='weight', task_mode='prune'):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        # Weight pruning
        # self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        # Channel Finetuning or Resume Pruning
        # self.popup_scores = Parameter(torch.Tensor(torch.Size([1,self.weight.shape[1]])))
        # Channel Pruning
        # self.popup_scores = Parameter(torch.Tensor(torch.Size([self.weight.shape[0],1])))

        self.prune_reg = prune_reg

        if self.prune_reg == 'weight':
            # Weight pruning or Filter Pruning
            self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        elif self.prune_reg == 'channel' and task_mode == 'finetune':
            # Channel Finetuning or Resume Pruning
            self.popup_scores = Parameter(torch.Tensor(torch.Size([1,self.weight.shape[1]])))
        elif self.prune_reg == 'channel' and task_mode == 'prune':
            # Channel Pruning
            self.popup_scores = Parameter(torch.Tensor(torch.Size([self.weight.shape[0],1])))
        else:
            raise NameError('prune_reg "{}" or task_mode "{}" are not correct!'.format(prune_reg, task_mode))

        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.w = 0
        # self.register_buffer('w', None)

    def set_prune_rate(self, global_k, k):
        self.k = k
        self.prune_k = k

    def forward(self, x):
        """ Unstructured Comparison
        remaining_weights = int(self.k * len(self.weight.flatten()))
        idx_same_top_weights_scores = list(
            set(torch.topk(self.weight.abs().flatten(), remaining_weights).indices.tolist()).intersection(
                set(torch.topk(self.popup_scores.abs().flatten(), remaining_weights).indices.tolist())))
        num_remaining_weights = len(idx_same_top_weights_scores)
        print(
            f"SubnetLinear: Number of same indices for scores and weights that are left after pruning: "
            f"{num_remaining_weights}. These are {float(num_remaining_weights / remaining_weights)} percent of the "
            f"weights kept.")
        """
        """ Structured Comparison
        remaining_filters = int(self.k * self.weight.shape[0])
        idx_same_top_weights_scores = list(set(
            torch.topk(torch.linalg.norm(self.weight.abs().reshape(self.weight.shape[0], -1), 1, dim=1),
                       remaining_filters).indices.tolist()).intersection(
            torch.topk(torch.linalg.norm(self.popup_scores.abs().reshape(self.popup_scores.shape[0], -1), 1, dim=1),
                       remaining_filters).indices.tolist()))
        num_remaining_filters = len(idx_same_top_weights_scores)
        print(
            f"SubnetLinear: Number of same indices for filters that are left after pruning using scores or weights : "
            f"{num_remaining_filters}. These are {float(num_remaining_filters / remaining_filters)} percent of the "
            f"filters kept.")
        """

        adj = GetSubnet.apply(self.popup_scores.abs(), self.prune_k, self.prune_reg)

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.linear(x, self.w, self.bias)

        return x
