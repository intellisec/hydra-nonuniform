import torch
import json
import numpy as np
import logging
import os
import yaml
import sys
import shutil
from distutils.dir_util import copy_tree
from utils.model import subnet_to_dense


def save_checkpoint(
    state, is_best, args, result_dir, filename="checkpoint.pth.tar", save_dense=False
):
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(result_dir, filename),
            os.path.join(result_dir, "model_best.pth.tar"),
        )

    if save_dense:
        state["state_dict"] = subnet_to_dense(state["state_dict"], args.k)
        torch.save(
            subnet_to_dense(state, args.k),
            os.path.join(result_dir, "checkpoint_dense.pth.tar"),
        )
        if is_best:
            shutil.copyfile(
                os.path.join(result_dir, "checkpoint_dense.pth.tar"),
                os.path.join(result_dir, "model_best_dense.pth.tar"),
            )


def create_subdirs(sub_dir):
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir, "checkpoint"))


def write_to_file(file, data, option):
    with open(file, option) as f:
        f.write(data)


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)


# ref:https://github.com/allenai/hidden-networks/blob/master/configs/parser.py
def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


# ref: https://github.com/allenai/hidden-networks/blob/master/args.py
def parse_configs_file(args):
    # get commands from command line
    override_args = argv_to_vars(sys.argv)

    # load yaml file
    args.configs = f'./configs/config-{args.arch}-{args.dataset.lower()}.yml'
    yaml_txt = open(args.configs).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.configs}")
    args.__dict__.update(loaded_yaml)

    # Compose exp_name
    exp_name = '_'.join([args.dataset.lower(), args.prune_reg, args.stg_id])
    args.__dict__.update({'exp_name': exp_name})

    # Adpat scaled_score_init switch:
    if args.exp_mode == 'prune':
        args.__dict__.update({'scaled_score_init': True})
    if args.exp_mode == 'finetune':
        args.__dict__.update({'scaled_score_init': False})

    # Define source net for finetune
    if args.exp_mode == 'finetune':
        if not args.evaluate:
            source_net = os.path.join(args.result_dir, args.exp_name, 'prune', 'latest_exp/checkpoint/checkpoint.pth.tar')
            args.__dict__.update({'source_net': source_net})
        else:
            source_net = os.path.join(args.result_dir, args.exp_name, 'finetune', 'latest_exp/checkpoint/model_best.pth.tar')
            args.__dict__.update({'source_net': source_net})
        print(f'Source_net: {args.source_net}')


def parse_prune_stg(args):
    ds_name = args.dataset
    hydra_dir = os.getcwd()
    strategy_f = '/'.join(hydra_dir.split('/')[:-1] + ['strategies/{}.json'.format(ds_name)])
    strategies = json.load(open(os.path.join('configs', strategy_f)))

    heracles_stg = strategies[args.arch][args.prune_reg][args.stg_id]
    prune_stg = [[], [], []]
    if args.arch == 'vgg16_bn':
        assert len(heracles_stg) == 16, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
        prune_stg[0] = heracles_stg[:13]
        prune_stg[1] = heracles_stg[13:]
        prune_stg[2] = []
    elif args.arch == 'resnet18':
        if args.prune_reg == 'channel':
            assert len(heracles_stg) == 18, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = heracles_stg[:17]
            prune_stg[1] = heracles_stg[17:]
            prune_stg[2] = list(np.take(heracles_stg, (5, 9, 13)))
        if args.prune_reg == 'weight':
            assert len(heracles_stg) == 21, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = heracles_stg[:5] + heracles_stg[6:10] + heracles_stg[11:15] + heracles_stg[16:20]
            prune_stg[1] = heracles_stg[20:]
            prune_stg[2] = list(np.take(heracles_stg, (5, 10, 15)))
    elif args.arch == 'wrn_28_4':
        if args.prune_reg == 'channel':
            assert len(heracles_stg) == 26, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = heracles_stg[:25]
            prune_stg[1] = heracles_stg[25:]
            prune_stg[2] = list(np.take(heracles_stg, (1, 9, 17)))
        if args.prune_reg == 'weight':
            assert len(heracles_stg) == 29, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = heracles_stg[:1] + heracles_stg[2:10] + heracles_stg[11:19] + heracles_stg[20:28]
            prune_stg[1] = heracles_stg[28:]
            prune_stg[2] = list(np.take(heracles_stg, (1, 10, 19)))
    elif args.arch == 'ResNet50':
        if args.prune_reg == 'channel':
            assert len(heracles_stg) == 50, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = heracles_stg[:49]
            prune_stg[1] = heracles_stg[49:]
            prune_stg[2] = list(np.take(heracles_stg, (1, 10, 22, 40)))
        if args.prune_reg == 'weight':
            assert len(heracles_stg) == 54, '! ! {}-{}-{} is invalid'.format(args.arch, args.prune_reg, args.stg_id)
            prune_stg[0] = heracles_stg[:1] + heracles_stg[2:11] + heracles_stg[12:24] + heracles_stg[25:42] + heracles_stg[42:53]
            prune_stg[1] = heracles_stg[53:]
            prune_stg[2] = list(np.take(heracles_stg, (1, 11, 24, 42)))
    else:
        raise NameError('Strategy check only supports vgg16_bn, resnet18, wrn_28_4, resnet50 but no "{}"'.format(args.arch))

    args.__dict__.update({'conv_k': prune_stg[0],
                          'fc_k': prune_stg[1],
                          'shortcut_k': prune_stg[2]})

    if args.exp_mode == 'prune':
        log_outdir = os.path.join(args.result_dir, args.exp_name)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger()
        logger.addHandler(
            logging.FileHandler(os.path.join(log_outdir, "heracles_strategy.log"), "a")
        )

        logger.info(
            '>> {} prune with target rate {} on model {}'.format(args.prune_reg.upper(), args.k, args.arch.upper()))
        logger.info('$  Prune-Strategy on SubnetConv:   {}'.format(prune_stg[0]))
        logger.info('$  Prune-Strategy on SubnetLinear: {}'.format(prune_stg[1]))
        logger.info('$  Prune-Strategy on ShortcutConv: {}'.format(prune_stg[2]))

    if args.evaluate:
        log_outdir = os.path.join(args.result_dir, args.exp_name)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger()
        logger.addHandler(
            logging.FileHandler(os.path.join(log_outdir, "heracles_adv_eval.log"), "a")
        )

        logger.info(
            '>> {} prune with target rate {} on model {}'.format(args.prune_reg.upper(), args.k, args.arch.upper()))
        logger.info('$  Prune-Strategy on SubnetConv:   {}'.format(prune_stg[0]))
        logger.info('$  Prune-Strategy on SubnetLinear: {}'.format(prune_stg[1]))
        logger.info('$  Prune-Strategy on ShortcutConv: {}'.format(prune_stg[2]))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def write_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.val, global_step)

