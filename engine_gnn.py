# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/24 18:18
# ide： PyCharm

import math
import torch

from typing import Iterable
from utils import misc, metric, engine_plugin


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: Iterable,
        epoch: int,
        args,
        lr_scheduler=None,
        logger=None,
        tensorboard_writer=None,
        wandb_run=None,
        **kwargs
):
    if args.amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model.train()
    criterion.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.update(lr=misc.get_lr(optimizer))
    header = '\033[0;34mEpoch [{}]\033[0m'.format(epoch)
    device = torch.device(args.device)

    disable = args.disable_engine_plugin == 'train' or args.disable_engine_plugin == 'all'
    plugin = engine_plugin.TrainEnginePlugin(args, tensorboard_writer, wandb_run, disable=disable)
    plugin.pre_process()  # TODO pass/modify parameters

    if args.amp:
        with torch.cuda.amp.autocast(enabled=args.amp):
            if args.model == 'aict':
                x_all, H, H_raw, x_pos, H_pos, H_raw_pos, x_neg, H_neg, H_raw_neg = model(kwargs['node_features'],
                                                                                          kwargs['train_pos'],
                                                                                          kwargs['train_neg'])
                probs_log = model.train_prediction(x_all, x_pos, x_neg, kwargs['train_pos'], kwargs['train_neg'])
                if args.model_only_x:
                    loss = criterion(x_all, H, H_raw, None, None, None, None, None, None, probs_log, kwargs['ys_float'])
                else:
                    loss = criterion(x_all, H, H_raw, x_pos, H_pos, H_raw_pos, x_neg, H_neg, H_raw_neg, probs_log,
                                     kwargs['ys_float'])
            elif args.model == 'slgnn_tdhnn':
                probs_log = model.train_prediction(kwargs['node_features'], kwargs['train_pos'], kwargs['train_neg'],
                                                   ex=kwargs['ex'])
                loss = criterion(probs_log, kwargs['ys_float'])
    else:
        if args.model == 'aict':
            x_all, H, H_raw, x_pos, H_pos, H_raw_pos, x_neg, H_neg, H_raw_neg = model(kwargs['node_features'],
                                                                                      kwargs['train_pos'],
                                                                                      kwargs['train_neg'])
            probs_log = model.train_prediction(x_all, x_pos, x_neg, kwargs['train_pos'], kwargs['train_neg'])
            if args.model_only_x:
                loss = criterion(x_all, H, H_raw, None, None, None, None, None, None, probs_log, kwargs['ys_float'])
            else:
                loss = criterion(x_all, H, H_raw, x_pos, H_pos, H_raw_pos, x_neg, H_neg, H_raw_neg, probs_log,
                                 kwargs['ys_float'])
        elif args.model == 'slgnn_tdhnn':
            probs_log = model.train_prediction(kwargs['node_features'], kwargs['train_pos'], kwargs['train_neg'],
                                               ex=kwargs['ex'])
            loss = criterion(probs_log, kwargs['ys_float'])

    plugin.process()

    if not math.isfinite(loss):
        raise ValueError("Loss is {}, stopping training".format(loss))

    if args.amp:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.zero_grad()
        loss.backward()

    if args.optim_max_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.optim_max_norm, norm_type=2)
    optimizer.step()

    metric_logger.update(loss=loss.item())
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    plugin.post_process()  # TODO pass/modify parameters

    if lr_scheduler is not None:
        lr_scheduler.step()

    metric_logger.synchronize_between_processes()
    print(header + ':' + '  ' + "Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return resstat


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        epoch: int,
        args,
        logger=None,
        tensorboard_writer=None,
        wandb_run=None,
        **kwargs
):
    model.eval()
    criterion.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = '\033[0;32mEval\033[0m'
    device = torch.device(args.device)

    disable = args.disable_engine_plugin == 'eval' or args.disable_engine_plugin == 'all'
    plugin = engine_plugin.EvalEnginePlugin(args, tensorboard_writer, wandb_run, disable=disable)
    plugin.pre_process()  # TODO pass/modify parameters

    (f1_micro_dir, f1_macro_dir, f1_weighted_dir, f1_binary_dir, auc_prob_dir, auc_label_dir, matrix_dir), (
        f1_micro_dou,
        f1_macro_dou,
        f1_weighted_dou,
        f1_binary_dou,
        auc_prob_dou,
        auc_label_dou, matrix_dou) = model.test_func(
        kwargs['node_features'], kwargs['train_pos'], kwargs['train_neg'], kwargs['test_pos_dir'],
        kwargs['test_neg_dir'], ex=kwargs['ex'])

    metric_logger.update(f1_micro_dir=f1_micro_dir.item(), f1_macro_dir=f1_macro_dir.item(),
                         f1_weighted_dir=f1_weighted_dir.item(),
                         f1_binary_dir=f1_binary_dir.item(), auc_prob_dir=auc_prob_dir.item(),
                         auc_label_dir=auc_label_dir.item(),
                         f1_micro_dou=f1_micro_dou.item(), f1_macro_dou=f1_macro_dou.item(),
                         f1_weighted_dou=f1_weighted_dou.item(),
                         f1_binary_dou=f1_binary_dou.item(), auc_prob_dou=auc_prob_dou.item(),
                         auc_label_dou=auc_label_dou.item())
    plugin.process()  # TODO pass/modify parameters

    plugin.post_process(matrix=matrix_dou)  # TODO pass/modify parameters

    metric_logger.synchronize_between_processes()

    print(header + ':' + '  ' + "Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    weights = {
        'epinions': {'f1_micro_dou': 0.944, 'f1_macro_dou': 0.8944, 'f1_weighted_dou': 0.9428, 'f1_binary_dou': 0.9668,
                     'auc_prob_dou': 0.9702, 'auc_label_dou': 0.8779},
        'slashdot': {'f1_micro_dou': 0.8783, 'f1_macro_dou': 0.8327, 'f1_weighted_dou': 0.8761, 'f1_binary_dou': 0.9201,
                     'auc_prob_dou': 0.9322, 'auc_label_dou': 0.8215},
        'bitcoinAlpha': {'f1_micro_dou': 0.9428, 'f1_macro_dou': 0.8361, 'f1_weighted_dou': 0.9422,
                         'f1_binary_dou': 0.9683, 'auc_prob_dou': 0.9508, 'auc_label_dou': 0.8288},
        'bitcoinOTC': {'f1_micro_dou': 0.9448, 'f1_macro_dou': 0.8899, 'f1_weighted_dou': 0.9441,
                       'f1_binary_dou': 0.9677, 'auc_prob_dou': 0.9687, 'auc_label_dou': 0.8795},
    }
    n = 0
    sum = 0
    for k, v in stats.items():
        if 'dou' in k:
            if False:
                sum += v * weights[args.model_dataset][k]
            else:
                sum += v
            n += 1
    stats['avg'] = sum / n
    print(header + ':' + '  ' + "Averaged stats:", stats['avg'])

    return stats
