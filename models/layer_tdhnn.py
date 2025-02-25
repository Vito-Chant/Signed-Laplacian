#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/06/22 16:09
# @Author  : Tao Chen
# @File    : layer_tdhnn.py

import torch
import torch.nn as nn
from .utils import SpecialSpmm
from .tdhnn import HGNN_conv
import os

act_func = nn.LeakyReLU(0.2)


# act_func = torch.nn.PReLU()


class SpMergeAttentionLayer(nn.Module):
    def __init__(self, dim_in,
                 dim_out,
                 num_edges,
                 alpha=0.2,
                 num_relations=2,
                 bias=True,
                 node_dropout=0.5,
                 att_dropout=0.5,
                 basis_att=True,
                 bias_hgnn=True, up_bound=0.95, low_bound=0.9,
                 min_num_edges=64, k_n=10, k_e=10
                 ):
        super(SpMergeAttentionLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.basis_att = basis_att

        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(1, dim_out)))
            nn.init.xavier_normal_(self.bias.data)

            # self.bias_pos = nn.Parameter(torch.zeros(size=(1, dim_out)))
            # nn.init.xavier_normal_(self.bias_pos.data)
            # self.bias_neg = nn.Parameter(torch.zeros(size=(1, dim_out)))
            # nn.init.xavier_normal_(self.bias_neg.data)

            self.add_bias = True
        else:
            self.add_bias = False

        if self.basis_att:
            self.num_relations = num_relations
            self.num_bases = self.num_relations
            self.basis = nn.Parameter(torch.Tensor(self.num_bases, dim_in, dim_out))
            self.att = nn.Parameter(torch.Tensor(self.num_relations, self.num_bases))
            nn.init.xavier_normal_(self.basis.data)
            nn.init.xavier_normal_(self.att.data)
        else:
            self.Wr = nn.Parameter(torch.Tensor(num_relations, dim_in, dim_out))
            nn.init.xavier_normal_(self.Wr.data)

        self.mapping_func = nn.Parameter(torch.zeros(size=(1, dim_out * 2)))
        nn.init.xavier_normal_(self.mapping_func.data)

        self.act_func = act_func
        # Sparse matrix multiplication function
        self.spmm = SpecialSpmm()
        # Dropout layer for node features
        self.dropout_node = nn.Dropout(node_dropout)
        # Dropout layer for attention values
        self.dropout_att = nn.Dropout(att_dropout)

        # self.hconv_1 = HGNN_conv(in_ft=dim_in, out_ft=dim_in, num_edges=num_edges, bias=bias_hgnn, up_bound=up_bound,
        #                          low_bound=low_bound, min_num_edges=min_num_edges, k_n=k_n, k_e=k_e)

        self.hconv_2_1 = HGNN_conv(in_ft=dim_out, out_ft=dim_out, num_edges=num_edges, bias=bias_hgnn,
                                   up_bound=up_bound,
                                   low_bound=low_bound, min_num_edges=min_num_edges, k_n=k_n, k_e=k_e)
        self.hconv_2_2 = HGNN_conv(in_ft=dim_out, out_ft=dim_out, num_edges=num_edges, bias=bias_hgnn,
                                   up_bound=up_bound,
                                   low_bound=low_bound, min_num_edges=min_num_edges, k_n=k_n, k_e=k_e)

        # self.hconv_3 = HGNN_conv(in_ft=dim_out, out_ft=dim_out, num_edges=num_edges, bias=bias_hgnn, up_bound=up_bound,
        #                          low_bound=low_bound, min_num_edges=min_num_edges, k_n=k_n, k_e=k_e)

    @staticmethod
    def reps_concatenation(src, dst):
        return torch.cat((src, dst), dim=1)

    def forward(self, node_reps, adj_pos, adj_neg, shape=None):
        if shape is None:
            n_row = node_reps.size()[0]
            n_col = n_row
        else:
            n_row, n_col = shape

        num_pos, num_neg = adj_pos.size()[1], adj_neg.size()[1]

        node_reps = self.dropout_node(node_reps)

        if self.basis_att:
            self.Wr = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
            # print("Wr.shape", self.Wr.shape)
            self.Wr = self.Wr.view(self.num_relations, self.dim_in, self.dim_out)
            # print("Wr.shape_view", self.Wr.shape)

        # TODO
        # node_reps, _, _ = self.hconv_1(node_reps)
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        h_pos, h_neg = torch.mm(node_reps, self.Wr[0]), torch.mm(node_reps, self.Wr[1])
        # print("h_pos.shape", h_pos.shape)
        # print("h_neg.shape", h_neg.shape)
        # TODO
        # ablation 4
        if os.environ['AB'] == '4':
            h_pos, _, _ = self.hconv_2_1(h_pos)
        elif os.environ['AB'] == '5':
            h_neg, _, _ = self.hconv_2_2(h_neg)
        else:
            h_pos, _, _ = self.hconv_2_1(h_pos)
            h_neg, _, _ = self.hconv_2_2(h_neg)

        h_pos_left, h_pos_right = h_pos[adj_pos[0, :], :], h_pos[adj_pos[1, :], :]
        # print("h_pos_left.shape", h_pos_left.shape)
        h_neg_left, h_neg_right = h_neg[adj_neg[0, :], :], h_neg[adj_neg[1, :], :]
        # print("h_pos_right.shape", h_pos_right.shape)

        sg_rep_pos = self.reps_concatenation(h_pos_left, h_pos_right)
        sg_rep_neg = self.reps_concatenation(h_neg_left, h_neg_right)
        sg_rep_all = torch.cat((sg_rep_pos, sg_rep_neg), dim=0)

        sg_coefficients = torch.sigmoid(self.act_func(self.mapping_func.mm(sg_rep_all.t()).squeeze()))
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # print("sg_coefficients.shape", sg_coefficients.shape)

        tensor_ones_col = torch.ones(size=(n_col, 1))
        tensor_ones_col = tensor_ones_col.to(adj_pos.device)
        adj_all = torch.cat((adj_pos, adj_neg), dim=1)

        edge_row_sum = self.spmm(adj_all, torch.ones_like(sg_coefficients), torch.Size([n_row, n_col]),
                                 tensor_ones_col) ** 0.5
        sym_normalize_coefficients = edge_row_sum[adj_all[0]] * edge_row_sum[adj_all[1]]
        sg_coefficients = sg_coefficients.div(sym_normalize_coefficients.squeeze())

        sg_coefficients = self.dropout_att(sg_coefficients)

        # ablation 3
        if os.environ['AB'] == '0' or os.environ['AB'] == '1' or os.environ['AB'] == '2':
            h_agg_pos = h_pos
            h_agg_neg = h_neg
        else:
            h_agg_pos = self.spmm(adj_pos, sg_coefficients[:num_pos], torch.Size([n_row, n_col]), h_pos)
            h_agg_neg = self.spmm(adj_neg, sg_coefficients[-num_neg:], torch.Size([n_row, n_col]), h_neg)

        # TODO
        # h_agg_pos, _, _ = self.hconv_2_1(h_agg_pos)
        # h_agg_neg, _, _ = self.hconv_2_2(h_agg_neg)

        output = h_agg_pos - h_agg_neg
        # TODO
        # output, _, _ = self.hconv_3(output)

        if self.add_bias:
            output = output + self.bias
            # h_agg_pos = h_agg_pos + self.bias_pos
            # h_agg_neg = h_agg_neg + self.bias_neg
        return output, h_agg_pos, h_agg_neg
