# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

import os
import time
import random
import datetime
import numpy as np
from collections import defaultdict, deque
from contextlib import contextmanager
from argparse import Action
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.init import xavier_uniform_, constant_


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        if d.shape[0] == 0:
            return 0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            # print(name, str(meter))
            # import ipdb;ipdb.set_trace()
            if meter.count > 0:
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        if logger is None:
            print_func = print
        else:
            print_func = logger.info

        i = 1
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header + ':',
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header + ':',
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        if print_freq == 0:
            # desc = header[:-1] if header.endswith(':') else header
            # for obj in tqdm(iterable, desc=desc, unit='b', ncols=80):
            for obj in tqdm(iterable, desc=header, unit='b', ncols=80):
                yield obj
        else:
            for obj in iterable:
                data_time.update(time.time() - end)
                yield obj

                iter_time.update(time.time() - end)
                if i % print_freq == 0 or i == len(iterable):
                    eta_seconds = iter_time.global_avg * (len(iterable) - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    if torch.cuda.is_available():
                        print_func(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                    else:
                        print_func(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)))
                i += 1
                end = time.time()
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print_func('{}:{}Total time: {} ({:.4f} s / it)'.format(
                header, self.delimiter, total_time_str, total_time / len(iterable)))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


@contextmanager
def torch_distributed_zero_first(rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if rank not in [-1, 0]:
        dist.barrier()
    yield
    if rank == 0:
        dist.barrier()
        dist.destroy_process_group()


def resolve_resume(resume):
    if type(resume) is not str:
        return resume
    if resume.lower() in ['none', 'null']:
        return None
    if resume.lower() in ['true', 'false']:
        return True if resume.lower() == 'true' else False
    return resume


def resolve_init_kwargs(args):
    model_init_kwargs = {}
    optim_init_kwargs = {}
    dataset_init_kwargs = {}
    for k, v in vars(args).items():
        if k.startswith('model_'):
            model_init_kwargs['_'.join(k.split('_')[1:])] = v
        if k.startswith('optim_'):
            optim_init_kwargs['_'.join(k.split('_')[1:])] = v
        if k.startswith('dataset_'):
            dataset_init_kwargs['_'.join(k.split('_')[1:])] = v

    return model_init_kwargs, optim_init_kwargs, dataset_init_kwargs


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val.lower() in ['none', 'null']:
            return None
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(',')]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def init_seeds(seed=42, cuda_deterministic=True):
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
        if cuda_deterministic:  # slower, more reproducible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.enabled = False
        else:  # slower, more reproducible
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    else:
        pass


def init_model_from_pretrain(model_without_module, pretrained_path=None, verbose=True, logger=None,
                             default_un_init=False):
    if verbose:
        if logger is None:
            print_func = print
        else:
            print_func = logger.info
    else:
        def void(*args, **kwargs):
            pass

        print_func = void

    model_dict = model_without_module.state_dict()
    if pretrained_path:
        # print_func('Initialized from pretrained model "{}"'.format(pretrained_path, ))
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        if pretrained_dict.__contains__('model'):
            pretrained_dict = pretrained_dict['model']
    else:
        pretrained_dict = {}

    new_state_dict = {k: v for k, v in pretrained_dict.items() if
                      (k in model_dict) and (v.size() == model_dict[k].size())}

    un_init_dict = {k: v for k, v in model_dict.items() if k not in new_state_dict}
    un_init_dict_keys = list(un_init_dict.keys())
    print_func('Initialized from pretrained model "{}"'.format(pretrained_path) + '\n' + "Un_init_dict_keys: " + str(
        un_init_dict_keys))

    if default_un_init:
        for k in un_init_dict_keys:
            new_state_dict[k] = torch.DoubleTensor(model_dict[k].size()).zero_()
            if 'weight' in k:
                if 'bn' in k:
                    print_func("{} init as: 1".format(k))
                    constant_(new_state_dict[k], 1)
                else:
                    try:
                        xavier_uniform_(new_state_dict[k])
                        print_func("{} init as: xavier".format(k))
                    except:
                        constant_(new_state_dict[k], 0)
                        print_func("{} init as: 0".format(k))
            elif 'bias' in k:
                print_func("{} init as: 0".format(k))
                constant_(new_state_dict[k], 0)
    else:
        new_state_dict.update(un_init_dict)

    model_without_module.load_state_dict(new_state_dict)


def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if not str(layer.__class__).startswith("<class 'torch.nn.modules"):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend(list(filter(lambda p: p.requires_grad, layer.parameters())))
            else:
                paras_wo_bn.extend(list(filter(lambda p: p.requires_grad, layer.parameters())))

    return paras_only_bn, paras_wo_bn


class EarlyStopping:
    def __init__(self, metric, init_score=None, better='small', start_step=0, patience=7, delta=0, verbose=True,
                 logger=None):
        """
        Args:
            patience (int): How long to wait after last time metric improved.
                            Default: 7
            verbose (bool): Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            logger: Default: None
        """
        if verbose:
            if logger is None:
                self.print_func = print
            else:
                self.print_func = logger.info
        else:
            def void(*args, **kwargs):
                pass

            self.print_func = void

        assert better in ['large', 'small']
        self.better = better
        if init_score is not None:
            self.init_score = init_score
        else:
            if better == 'small':
                self.init_score = float('inf')
            else:
                self.init_score = -float('inf')
        self.best_score = self.init_score
        self.metric_name = metric

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.start_step = start_step
        self.step_cnt = 0

    def step(self, score):
        if self.step_cnt < self.start_step:
            self.step_cnt += 1
            return
        if self.isbetter(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.print_func('Early stopping at step {}'.format(self.step_cnt))
        self.step_cnt += 1

    def isbetter(self, score):
        if self.better == 'large':
            return score > self.best_score + self.delta
        if self.better == 'small':
            return score < self.best_score - self.delta


def get_layer_computation(model):
    """
    获取模型每一层的计算量（这里使用参数数量作为计算量的近似）
    """
    layer_computations = []
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            # 对于卷积层，计算参数量
            num_params = sum(p.numel() for p in layer.parameters())
            layer_computations.append(num_params)
        elif isinstance(layer, nn.Linear):
            # 对于全连接层，计算参数量
            num_params = sum(p.numel() for p in layer.parameters())
            layer_computations.append(num_params)
        else:
            # 对于其他层，暂时使用0来表示
            layer_computations.append(0)

    return layer_computations


def distribute_model(model, device_list):
    """
    自动将模型的不同部分放到不同的GPU上 (模型并行)

    Args:
        model: 要进行并行处理的模型 (nn.Module)
        device_list: 一个列表，包含多个GPU设备的ID（例如：[0, 1, 2]）

    Returns:
        返回分布式模型
    """
    if len(device_list) < 2:
        raise ValueError("至少需要两块GPU进行模型并行！")

    # 获取模型的所有层
    num_gpus = len(device_list)
    layer_computations = get_layer_computation(model)

    # 根据计算量排序层
    sorted_layer_indices = np.argsort(layer_computations)

    # 初始化每个GPU的负载
    gpu_loads = [0] * num_gpus
    devices = [torch.device(f'cuda:{device}') for device in device_list]

    # 用于保存每个GPU上的子模型
    module_list = [[] for _ in range(num_gpus)]

    for idx in sorted_layer_indices:
        # 找到当前层对应的GPU
        min_load_gpu_idx = np.argmin(gpu_loads)

        # 将当前层分配给负载最小的GPU
        module_list[min_load_gpu_idx].append(list(model.children())[idx].to(devices[min_load_gpu_idx]))

        # 更新GPU负载（这里我们简单地按参数量更新，当然你也可以考虑更细致的计算量）
        gpu_loads[min_load_gpu_idx] += layer_computations[idx]

    class ModelParallel(type(model)):
        def __init__(self, modules):
            self.device_list = [torch.device(f'cuda:{device}') for device in device_list]
            self.modules = nn.ModuleList(modules)
            self.__dict__.update(model.__dict__)

        def forward(self, x):
            for i, module in enumerate(self.modules):
                x = module(x)
            return x

    optimized_modules = []
    for i in range(num_gpus):
        optimized_modules.append(nn.Sequential(*module_list[i]))

    # 创建并返回一个并行模型
    return ModelParallel(optimized_modules)
