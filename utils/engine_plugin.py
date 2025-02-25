# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/7/14 15:32 
# ide： PyCharm

import torch
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

from utils import metric


class TrainEnginePlugin():
    def __init__(self, args, tensorboard_writer, wandb_run, disable=False):
        self.tensorboard_writer = tensorboard_writer
        self.wandb_run = wandb_run
        self.args = args
        self.disable = disable

    def pre_process(self, **kwargs):
        if not self.disable:
            pass

    def process(self, **kwargs):
        if not self.disable:
            pass

    def post_process(self, **kwargs):
        if not self.disable:
            pass


class EvalEnginePlugin():
    def __init__(self, args, tensorboard_writer=None, wandb_run=None, disable=False):
        self.tensorboard_writer = tensorboard_writer
        self.wandb_run = wandb_run
        self.args = args
        self.disable = disable

        self.tb_args = {'val_embedding_freq': 5,
                        'val_embedding_layer': None,  # set None to forbid log embedding
                        }
        self.wandb_args = {}

        self.embedding = None
        self.metadata = None
        self.handle = None
        self.confusion_matrix = metric.ConfusionMatrix(2, ['Positive', 'Negative'])

    def pre_process(self, **kwargs):
        if not self.disable:
            pass

    def process(self, **kwargs):
        if not self.disable:
            pass

    def post_process(self, **kwargs):
        if not self.disable:
            if self.wandb_run is not None:
                self.confusion_matrix.enforce(kwargs['matrix'])
                # cm = wandb.plot.confusion_matrix(y_true=self.confusion_matrix.labels, preds=self.confusion_matrix.preds)
                # self.wandb_run.log({"Confusion matrix": cm})

                cm = self.confusion_matrix.plot()
                self.wandb_run.log({"Confusion matrix": cm})
                plt.close(cm)
            return self.confusion_matrix.summary()
