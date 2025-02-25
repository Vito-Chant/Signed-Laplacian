#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/07/06 20:28
# @Author  : Tao Chen
# @File    : gtrick.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_sparse import SparseTensor

class CommonNeighbors:
    def __init__(self, edge_index, edge_attr=None, batch_size=64, sparse_sizes=None) -> None:
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1])

        self.edge_index = edge_index
        self.A = SparseTensor.from_edge_index(
            edge_index, edge_attr=edge_attr, sparse_sizes=sparse_sizes).t()

        self.batch_size = batch_size

    def __call__(self, edges):
        idx_loader = DataLoader(range(edges.shape[0]), self.batch_size)
        cn = []

        print('Calculating common neighbors as edge feature...')
        for idx in tqdm(idx_loader):
            src, dst = edges[idx, 0], edges[idx, 1]
            cn.append(torch.sum(self.A[src].to_dense()
                                * self.A[dst].to_dense(), 1))

        cn = torch.cat(cn, 0)
        return cn.view(-1, 1)
