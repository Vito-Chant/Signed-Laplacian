# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/24 18:22 
# ide： PyCharm

import torch
from utils import registry
from .resnet import *
from .tdhnn import HGNN_link_prediction
from .aict_loss import AICTLoss
from .slgnn_tdhnn import SLGNN

MODEL_BUILD_FUNCS = registry.Registry('model and criterion build functions')
'''
The return value of a build function must be one of two ways:
    (1) return model, criterion
    (2) return model, (criterion_train, criterion_eval)
For way (2), if criterion_eval is assigned to "None", this run will not execute “evaluate”
'''


@MODEL_BUILD_FUNCS.register_with_name(module_name='resnet')
def build_resnet(arch='resnet18', num_classes=7, img_channels=3, pretrained=False):
    model_handle = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'resnext50_32x4d': resnext50_32x4d,
        'resnext101_32x8d': resnext101_32x8d,
        'wide_resnet50_2': wide_resnet50_2,
        'wide_resnet101_2': wide_resnet101_2
    }
    model = model_handle[arch](pretrained, num_classes=num_classes, img_channels=img_channels)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion_train = loss_fn_1
    # criterion_eval = loss_fn_2

    return model, criterion
    # return model, (criterion_train, criterion_eval)


@MODEL_BUILD_FUNCS.register_with_name(module_name='aict')
def build_aict(dataset, dataset_fold, feature_dim, dropout=0.5, in_dim=64, hid_dim=64, num_edges=100,
               alpha=0.2, node_dropout=0.5, att_dropout=0.5, nheads=4, conv_number=1, transfer=1, up_bound=0.95,
               low_bound=0.9, min_num_edges=64, k_n=10, k_e=10, self_attention_heads=4, namuda=30, namuda2=10,
               only_x=False, others_loss=False):
    model = HGNN_link_prediction(dropout=dropout, in_dim=in_dim, hid_dim=hid_dim, num_edges=num_edges, alpha=alpha,
                                 node_dropout=node_dropout, att_dropout=att_dropout, nheads=nheads,
                                 conv_number=conv_number, transfer=transfer, up_bound=up_bound, low_bound=low_bound,
                                 min_num_edges=min_num_edges, k_n=k_n, k_e=k_e,
                                 self_attention_heads=self_attention_heads, only_x=only_x)
    criterion_train = AICTLoss(namuda=namuda, namuda2=namuda2, others_loss=others_loss)
    criterion_eval = torch.nn.NLLLoss()

    return model, (criterion_train, criterion_eval)


@MODEL_BUILD_FUNCS.register_with_name(module_name='slgnn_tdhnn')
def build_slgnn_tdhnn(dataset, dataset_fold, feature_dim, in_dim=64, hid_dim=64, num_edges=100,
                      alpha=0.2, node_dropout=0.5, att_dropout=0.5, nheads=4, up_bound=0.95,
                      low_bound=0.9, min_num_edges=64, k_n=10, k_e=10, bias_hgnn=True, common_neighbors=False,
                      nr=0.0):
    model = SLGNN(num_feats=in_dim,
                  alpha=alpha,
                  node_dropout=node_dropout,
                  att_dropout=att_dropout,
                  nheads=nheads,
                  dim_hidden=hid_dim,
                  bias_hgnn=bias_hgnn, up_bound=up_bound, low_bound=low_bound,
                  min_num_edges=min_num_edges, k_n=k_n, k_e=k_e, num_edges=num_edges, common_neighbors=common_neighbors)
    criterion_train = torch.nn.NLLLoss()
    criterion_eval = torch.nn.NLLLoss()

    return model, (criterion_train, criterion_eval)
