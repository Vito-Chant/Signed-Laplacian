#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/05/31 10:12
# @Author  : Tao Chen
# @File    : sweep_optuna.py

import faulthandler

# 在import之后直接添加以下启用代码即可
faulthandler.enable()
# 后边正常写你的代码
import os
import sys
import yaml
import shutil
import optuna
import argparse
import random
import json  # Added for parameter serialization
import numpy as np

from main_gnn import main as _main
from main_gnn import get_args_parser
from functools import partial
from typing import Optional, List, Any, Dict
from copy import deepcopy
from collections import defaultdict


def default_args(args):
    default_args_dict = {
        'eval': False,
        'resume': None,
        'tags': None,
        'variant_file': None,
        'options': None,
        'force_override': True
    }

    if args.parallel_mode == 'ddp':
        raise ValueError('ddp is not supported yet')

    for k, v in default_args_dict.items():
        setattr(args, k, v)


def load_existing_trials(log_path: str) -> Dict[str, float]:
    """
    Load existing trials from the sweep log file.

    Args:
        log_path (str): Path to the sweep_log.txt file.

    Returns:
        Dict[str, float]: A dictionary mapping serialized hyperparameters to their metrics.
    """
    existing = {}
    if not os.path.exists(log_path):
        return existing

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Iterate through lines to find hyperparameters and their metrics
    for i in range(len(lines)):
        if lines[i].startswith('Trial value:'):
            # Extract the metric
            metric_line = lines[i].strip()
            metric = metric_line.split('Trial value:')[1].strip()

            # The next line should contain hyperparameters
            if i + 1 < len(lines) and lines[i + 1].startswith('Trial hyperparameters:'):
                params_line = lines[i + 1].strip()
                params_str = params_line.split('Trial hyperparameters:')[1].strip()
                params = json.loads(params_str.replace("'", "\""))  # Convert to valid JSON
                # Serialize the parameters to a sorted JSON string for consistent comparison
                params_serialized = json.dumps(params, sort_keys=True)
                existing[params_serialized] = metric

    return existing


def objective(trial, args, space, existing_trials: Optional[Dict[str, float]] = None):
    print(f"Trial {trial.number} is starting.\n")
    with open(args.sweep_log, 'a') as f:
        f.write(f"Trial {trial.number} is starting.\n")
    _args = deepcopy(args)
    _args.name = '[sweep_optuna]{}/{}'.format(_args.name, trial.number)
    _args.group = '[sweep_optuna]{}'.format(_args.name)
    default_args(_args)
    for k, v in space.items():
        setattr(_args, k, SweepConfig.suggest_handle(trial, v['type'])(k, **v['meta']))

    # Serialize current hyperparameters for comparison
    current_params = {k: getattr(_args, k) for k in space.keys()}
    params_serialized = json.dumps(current_params, sort_keys=True)

    if args.sweep_resume and existing_trials is not None:
        if params_serialized in existing_trials:
            stored_metric = existing_trials[params_serialized]
            print(f"Trial {trial.number} has been previously evaluated. Skipping execution.")
            with open(args.sweep_log, 'a') as f:
                info = f"Trial {trial.number} skipped. Stored value: {stored_metric}\n"
                f.write(info)
            return stored_metric

    metrics = []
    best_stats_list = defaultdict(list)
    _args.model_min_num_edges = int(_args.model_min_num_edges * _args.model_num_edges)
    _args.model_k_e = int(_args.model_k_e * _args.model_min_num_edges)
    for i in range(1):
        setattr(_args, 'model_dataset_fold', i)
        setattr(_args, 'name', '[sweep_optuna]{}/{}'.format(args.name, trial.number) + f'_{i}')
        _metric, best_stats = _main(_args)
        metrics.append(_metric)
        for k, v in best_stats.items():
            if 'dou' in k:
                best_stats_list[k].append(v)

    metric = sum(metrics) / len(metrics)
    std = np.std(np.array(metrics), ddof=0).item()

    with open(_args.sweep_log, 'a') as f:
        info = f'Trial value: {metric:.4f}±{std:.4f}' + '\n' + 'Trial hyperparameters: {}'.format(
            trial.params) + '\n' + 'Others: '
        for k, v in best_stats_list.items():
            v = np.array(v)
            mean = np.mean(v).item()
            std = np.std(v, ddof=0).item()
            info += f'{k}: {mean:.4f}±{std:.4f}\t'
        info += '\n'

        f.write(info)

    return metric


def callback(study, trial, **kwargs):
    # 每次试验结束后被调用
    best_trial = study.best_trial
    with open(kwargs['sweep_log'], 'a') as f:
        f.write('Current best is trial {} with value: {}\n\n'.format(best_trial.number, best_trial.value))


class SweepConfig(object):
    def __init__(self, space: dict, sampler: str = 'TPE', count: int = None, **kwargs):
        self.count = count
        self.space = space
        self.sampler_name = sampler
        if sampler.lower() == 'tpe':
            self.sampler = optuna.samplers.TPESampler(**kwargs)
        elif sampler.lower() == 'random':
            self.sampler = optuna.samplers.RandomSampler(**kwargs)
        elif sampler.lower() == 'cmaes':
            self.sampler = optuna.samplers.CmaEsSampler(**kwargs)
        elif sampler.lower() == 'grid':
            param_grid = {}
            self.count = 1
            for k, v in space.items():
                assert v['type'] == 'categorical'
                param_grid[k] = v['meta']['choices']
                self.count *= len(v['meta']['choices'])
            self.sampler = optuna.samplers.GridSampler(param_grid, **kwargs)
        else:
            raise ValueError('Unsupported sampler: {}'.format(sampler))

    def __str__(self):
        _dict = {'sampler': self.sampler_name,
                 'space': self.space,
                 'count': self.count, }

        return yaml.dump(_dict)[:-1]

    @staticmethod
    def suggest_handle(trial, suggest_type):
        if suggest_type == 'float':
            return trial.suggest_float
        elif suggest_type == 'int':
            return trial.suggest_int
        elif suggest_type == 'categorical':
            return trial.suggest_categorical
        else:
            raise ValueError('Unsupported suggest type: {}'.format(suggest_type))

    @staticmethod
    def float(low: float, high: float, step: Optional[float] = None, log: bool = False):
        return {'type': 'float', 'meta': {'low': low, 'high': high, 'step': step, 'log': log}}

    @staticmethod
    def int(low: int, high: int, step: int = 1, log: bool = False):
        return {'type': 'int', 'meta': {'low': low, 'high': high, 'step': step, 'log': log}}

    @staticmethod
    def categorical(choices: List[Any]):
        return {'type': 'categorical', 'meta': {'choices': choices}}


def sweep(sweep_config):
    from build_config import config

    config()
    parser = argparse.ArgumentParser('Model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.resume:
        setattr(args, 'sweep_resume', True)
    else:
        setattr(args, 'sweep_resume', False)
    default_args(args)

    if os.path.exists(f'./runs/[sweep_optuna]{args.name}'):
        if not args.sweep_resume:
            q = input('\033[1;31mA sweep with the same name "{}" has already existed, '
                      'whether to override [y/n]: \033[0m'.format(args.name))
            if q.lower() == 'y':
                shutil.rmtree(f'./runs/[sweep_optuna]{args.name}')
            else:
                sys.exit()
        else:
            print("Resuming the sweep from existing runs.")

    sweep_dir = './runs/[sweep_optuna]{}'.format(args.name)
    os.makedirs(sweep_dir, exist_ok=True)
    shutil.copy("./config/config.yml", sweep_dir)
    args.config_file = os.path.join(sweep_dir, 'config.yml')
    args.sweep_log = os.path.join(sweep_dir, 'sweep_log.txt')

    print('Full sweep configuration:\n' + str(sweep_config))
    with open(os.path.join(sweep_dir, 'sweep_config.yml'), "w") as f:
        f.write(str(sweep_config))
    print(f"Full sweep configuration saved to '{os.path.join(sweep_dir, 'sweep_config.yml')}'")

    # Load existing trials if resuming
    existing_trials = {}
    if args.sweep_resume:
        existing_trials = load_existing_trials(args.sweep_log)
        print(f"Loaded {len(existing_trials)} existing trials for resuming.")
    print(existing_trials)

    # Create Optuna study with a unique storage if needed
    study = optuna.create_study(
        study_name=args.name,
        sampler=sweep_config.sampler,
        direction="maximize" if args.better == 'large' else "minimize",
        load_if_exists=True  # This allows resuming studies if using a persistent storage
    )

    # Define the objective with existing_trials passed as a parameter
    study.optimize(
        partial(objective, args=args, space=sweep_config.space, existing_trials=existing_trials),
        n_trials=sweep_config.count,
        callbacks=[partial(callback, sweep_log=args.sweep_log)]
    )

    best_trial = study.best_trial
    with open(args.sweep_log, 'a') as f:
        f.write(f'Best is trial {best_trial.number} with value: {best_trial.value}\n')
        f.write(f"Best hyperparameters is: {json.dumps(study.best_params, sort_keys=True)}\n")


if __name__ == '__main__':
    # sweep_config = SweepConfig(
    #     space={
    #         'model_alpha': SweepConfig.float(0.1, 0.9),
    #         'model_att_dropout': SweepConfig.float(0.1, 0.9),
    #         'model_dropout': SweepConfig.float(0.1, 0.9),
    #         'model_hid_dim': SweepConfig.categorical([64, 128]),
    #         'model_k_e': SweepConfig.float(0.1, 1.0),
    #         'model_k_n': SweepConfig.int(1, 100),
    #         'model_low_bound': SweepConfig.float(0.1, 0.9),
    #         'model_node_dropout': SweepConfig.float(0.1, 0.9),
    #         'model_num_edges': SweepConfig.int(2, 1000),
    #         'model_up_bound': SweepConfig.float(0.1, 0.9),
    #         'model_nheads': SweepConfig.categorical([2, 4]),
    #         'model_min_num_edges': SweepConfig.float(0.1, 0.9)
    #     },
    #     count=100,
    #     sampler='tpe'
    # )

    # sweep_config = SweepConfig(
    #     space={
    #         'model_alpha': SweepConfig.categorical([0.7]),
    #         'model_att_dropout': SweepConfig.categorical([0.1]),
    #         'model_dropout': SweepConfig.categorical([0.1]),
    #         'model_hid_dim': SweepConfig.categorical([128]),
    #         'model_k_e': SweepConfig.categorical([0.3]),
    #         'model_k_n': SweepConfig.categorical([100]),
    #         'model_low_bound': SweepConfig.categorical([0.5]),
    #         'model_node_dropout': SweepConfig.categorical([0.3]),
    #         'model_num_edges': SweepConfig.categorical([500]),
    #         'model_up_bound': SweepConfig.categorical([0.35]),
    #         'model_nheads': SweepConfig.categorical([2]),
    #         'model_min_num_edges': SweepConfig.categorical([0.3])
    #     },
    #     sampler='grid'
    # )

    # # slgnn_tdhnn
    # sweep_config = SweepConfig(
    #     space={
    #         'model_alpha': SweepConfig.categorical([0.17957915079389092]),
    #         'model_att_dropout': SweepConfig.categorical([0.8226695900888544]),
    #         'model_hid_dim': SweepConfig.categorical([128]),
    #         'model_k_e': SweepConfig.categorical([0.9391040481545969]),
    #         'model_k_n': SweepConfig.categorical([64]),
    #         'model_low_bound': SweepConfig.categorical([0.5401179340035372]),
    #         'model_node_dropout': SweepConfig.categorical([0.1696518982395433]),
    #         'model_num_edges': SweepConfig.categorical([539]),
    #         'model_up_bound': SweepConfig.categorical([0.6048302578488381]),
    #         'model_nheads': SweepConfig.categorical([4]),
    #         'model_min_num_edges': SweepConfig.categorical([0.7275300897129856])
    #     },
    #     sampler='grid'
    # )
    import os

    os.environ['AB'] = '-1'
    sweep_config = SweepConfig(
        space={
            'model_alpha': SweepConfig.categorical([0.2]),
            'model_att_dropout': SweepConfig.categorical([0.8]),

            'model_hid_dim': SweepConfig.categorical([128]),  # 超边特征矩阵的维度, 128
            # 'model_hid_dim': SweepConfig.categorical([32, 64, 256]),  # 超边特征矩阵的维度

            'model_k_e': SweepConfig.categorical([0.9]),  # 每个节点在超图中可以连接的超边数量
            # 'model_k_e': SweepConfig.categorical([0.3, 0.6, 1.0]),  # 每个节点在超图中可以连接的超边数量

            'model_k_n': SweepConfig.categorical([64]),  # 更新超边特征时考虑的节点数量 64
            # 'model_k_n': SweepConfig.categorical([16, 32, 128]),  # 更新超边特征时考虑的节点数量

            'model_num_edges': SweepConfig.categorical([500]),  # 开始时用于构建超图的超边数量
            # 'model_num_edges': SweepConfig.categorical([100, 250, 1000]),  # 开始时用于构建超图的超边数量

            'model_nheads': SweepConfig.categorical([4]),  # 4
            # 'model_nheads': SweepConfig.categorical([1, 2, 8]),  #

            'model_min_num_edges': SweepConfig.categorical([0.7]),  # 最少超边数
            # 'model_min_num_edges': SweepConfig.categorical([0.2, 0.5, 1.0]),  # 最少超边数

            'model_low_bound': SweepConfig.categorical([0.5]),  # 调整超边数量的饱和度得分的下限阈值
            'model_up_bound': SweepConfig.categorical([0.6]),  # 饱和度得分的上限阈值，用于控制超边数量的增加
            'model_node_dropout': SweepConfig.categorical([0.2]),
            'model_dataset': SweepConfig.categorical(['bitcoinAlpha']),
            # 'bitcoinAlpha', 'bitcoinOTC', 'epinions', 'slashdot'
            # 'seed': SweepConfig.categorical([42])
            'seed': SweepConfig.categorical(random.sample(range(10000, 40000), 2)),
            'model_nr': SweepConfig.categorical([0]),
        },
        sampler='grid'
    )

    # sweep_config = SweepConfig(
    #     space={
    #         'model_alpha': SweepConfig.float(0.1, 0.9),
    #         'model_att_dropout': SweepConfig.float(0.1, 0.9),
    #         # 'model_hid_dim': SweepConfig.categorical([64]),
    #         'model_hid_dim': SweepConfig.categorical([128]),
    #         'model_k_e': SweepConfig.float(0.1, 1.0),
    #         'model_k_n': SweepConfig.int(1, 100),
    #         'model_low_bound': SweepConfig.float(0.1, 0.9),
    #         'model_node_dropout': SweepConfig.float(0.1, 0.9),
    #         'model_num_edges': SweepConfig.int(2, 1000),
    #         'model_up_bound': SweepConfig.float(0.1, 0.9),
    #         # 'model_nheads': SweepConfig.categorical([2]),
    #         'model_nheads': SweepConfig.categorical([4]),
    #         'model_min_num_edges': SweepConfig.float(0.1, 0.9)
    #     },
    #     count=100,
    #     sampler='tpe'
    # )

    sweep(sweep_config)

# python sweep_optuna.py -m slgnn_tdhnn -o adagrad -e 1500 --metric f1_micro_dou --disable_wandb --option model_dataset=bitcoinAlpha  -g 5 -n hp_k_e
