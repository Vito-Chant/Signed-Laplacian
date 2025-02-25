#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/06/22 16:51
# @Author  : Tao Chen
# @File    : slgnn_tdhnn.py
from __future__ import print_function

import torch
from sklearn import metrics
import numpy as np
import torch.nn as nn
import warnings
from .aggregators_tdhnn import LayerAggregator
from .tdhnn import HGNN_conv

warnings.filterwarnings("ignore")

act_func = nn.ReLU()


# act_func = nn.PReLU()


class LinkClassifierMLP(nn.Module):
    def __init__(self, dim, dropout=0.0, common_neighbors=False):
        super().__init__()
        _dim = dim
        if common_neighbors:
            dim *= 5
        else:
            dim *= 4
        hidden1, hidden2, output = int(dim // 2), int(dim // 4), 1  # final_dim * 4 => final_dim * 2 => final_dim => 1
        activation = act_func
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden1),
            nn.Dropout(dropout),
            activation,
            nn.Linear(hidden1, hidden2),
            nn.Dropout(dropout),
            activation,
            nn.Linear(hidden2, output)
        )
        self.common_neighbors = common_neighbors
        if self.common_neighbors:
            self.common_neighbors_encoder = torch.nn.Linear(1, _dim)

    def forward(self, x, ex=None):
        if self.common_neighbors:
            # x = x + self.common_neighbors_encoder(ex)
            x = torch.cat([x, self.common_neighbors_encoder(ex)], dim=1)
        res = self.layers(x)
        res = torch.clamp(res, -1e10, 1e10)
        return torch.sigmoid(res)


class LinkRepresentation(nn.Module):
    def __init__(self, embed_dim, fusion_hidden_dim, fusion_output_dim, dropout=0.5):
        """
        embed_dim: 源节点或目标节点的原始嵌入维度
        融合后拼接的维度为 4 * embed_dim，
        fusion_hidden_dim: 融合层的隐藏层维度
        fusion_output_dim: 最终输出的链接表示维度
        """
        super(LinkRepresentation, self).__init__()
        # 定义一个融合层，将拼接后的特征转换为更紧凑的表示
        self.fusion_layer = nn.Sequential(
            nn.Linear(4 * embed_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_output_dim),
            nn.ReLU()
        )

    def forward(self, src_emb, dst_emb):
        """
        src_emb: [batch_size, embed_dim] 源节点嵌入
        dst_emb: [batch_size, embed_dim] 目标节点嵌入
        返回：
            link_rep: [batch_size, fusion_output_dim] 链接表示
        """
        # 计算差值和逐元素乘积
        diff = src_emb - dst_emb
        prod = src_emb * dst_emb
        # 拼接原始嵌入、差值以及乘积信息
        combined = torch.cat([src_emb, dst_emb, diff, prod], dim=1)
        # 通过融合层获得最终表示
        link_rep = self.fusion_layer(combined)
        return link_rep


class SLGNN(nn.Module):
    def __init__(self,
                 num_feats,
                 alpha=0.2,
                 node_dropout=0.5,
                 att_dropout=0.5,
                 nheads=4,
                 dim_hidden=64,
                 bias_hgnn=True, up_bound=0.95, low_bound=0.9,
                 min_num_edges=64, k_n=10, k_e=10, num_edges=100,
                 common_neighbors=False
                 ):
        super(SLGNN, self).__init__()
        # -----------------------------------------------
        dim_in_layer1, dim_out_layer1, nheads1 = num_feats, dim_hidden, nheads
        dim_in_layer2, dim_out_layer2, nheads2 = dim_out_layer1 * nheads1, dim_hidden, nheads
        dim_out_final = dim_out_layer2
        # -----------------------------------------------
        self.agg1 = LayerAggregator(dim_in=dim_in_layer1, dim_out=dim_out_layer1, nheads=nheads1, alpha=alpha,
                                    node_dropout=node_dropout, att_dropout=att_dropout, bias_hgnn=bias_hgnn,
                                    up_bound=up_bound, low_bound=low_bound,
                                    min_num_edges=min_num_edges, k_n=k_n, k_e=k_e, num_edges=num_edges)
        self.agg2 = LayerAggregator(dim_in=dim_in_layer2, dim_out=dim_out_layer2, nheads=nheads2, alpha=alpha,
                                    node_dropout=node_dropout, att_dropout=att_dropout, last_layer=True,
                                    bias_hgnn=bias_hgnn, up_bound=up_bound, low_bound=low_bound,
                                    min_num_edges=min_num_edges, k_n=k_n, k_e=k_e, num_edges=num_edges)
        # -----------------------------------------------
        # if common_neighbors:
        #     self.link_rep_dimensions = dim_out_final * 5
        # else:
        #     self.link_rep_dimensions = dim_out_final * 4
        self.mlp = LinkClassifierMLP(dim_hidden, common_neighbors=common_neighbors)
        #
        # self.hconv_2_1 = HGNN_conv(in_ft=dim_in_layer1, out_ft=dim_out_layer1 * nheads1, num_edges=num_edges,
        #                            bias=bias_hgnn, up_bound=up_bound, low_bound=low_bound, min_num_edges=min_num_edges,
        #                            k_n=k_n, k_e=k_e)
        # self.hconv_2_2 = HGNN_conv(in_ft=dim_in_layer2, out_ft=dim_out_layer2, num_edges=num_edges,
        #                            bias=bias_hgnn, up_bound=up_bound, low_bound=low_bound, min_num_edges=min_num_edges,
        #                            k_n=k_n, k_e=k_e)
        self.fusion = LinkRepresentation(embed_dim=dim_out_final, fusion_hidden_dim=dim_out_final * 2,
                                         fusion_output_dim=dim_hidden * 4, dropout=0.5)

    def forward(self, node_reps, adj_pos, adj_neg):
        adj_pos = adj_pos.to(dtype=torch.long)
        adj_neg = adj_neg.to(dtype=torch.long)
        ######这里其实也有点问题，两个agg都输出了x_pos,x_neg，agg2的输入事agg1的x，但事agg1的x_pos,x_neg没用上，
        # x1, x_pos, x_neg = self.agg1(node_reps, adj_pos, adj_neg)
        # x2, _, _ = self.hconv_2_1(node_reps)
        # # x = x1 + x2
        # x1, x_pos, x_neg = self.agg2(x1, adj_pos, adj_neg)
        # x2, _, _ = self.hconv_2_2(x2)
        # # x = x1 + x2
        # x = x1

        x, x_pos, x_neg = self.agg1(node_reps, adj_pos, adj_neg)
        x, x_pos, x_neg = self.agg2(x, adj_pos, adj_neg)

        return x  # , x_pos, x_neg

    def train_prediction(self, node_reps, adj_pos, adj_neg, ex=None):
        x = self.forward(node_reps, adj_pos, adj_neg)

        train_i_index = torch.cat((adj_pos[0], adj_neg[0]), dim=0)
        train_j_index = torch.cat((adj_pos[1], adj_neg[1]), dim=0)
        link_representations = self.generate_link_representation(x, train_i_index, train_j_index)
        positive_prob = self.mlp(link_representations, ex=ex)
        negative_prob = 1 - positive_prob
        probs = torch.cat((negative_prob, positive_prob), dim=1)
        probs = torch.clamp(probs, 1e-10, 1)
        probs_log = torch.log(probs)

        return probs_log

    # @staticmethod  # TODO
    def generate_link_representation(self, reps, src, dst):
        src_emb, dst_emb = reps[src], reps[dst]
        # link_rep = torch.cat((src_emb, dst_emb, src_emb - dst_emb, src_emb * dst_emb), dim=1)
        link_rep = self.fusion(src_emb, dst_emb)
        return link_rep

    @staticmethod
    def evaluation(test_y, pred_p, name=""):
        pred = np.argmax(pred_p, axis=1)
        test_y = np.array(test_y)
        f1_macro = metrics.f1_score(test_y, pred, average='macro')
        f1_micro = metrics.f1_score(test_y, pred, average='micro')
        f1_weighted = metrics.f1_score(test_y, pred, average='weighted')
        f1_binary = metrics.f1_score(test_y, pred, average='binary')
        auc_prob = metrics.roc_auc_score(test_y, pred_p[:, 1])
        auc_label = metrics.roc_auc_score(test_y, pred)
        matrix = metrics.confusion_matrix(test_y, pred)

        # print(metrics.confusion_matrix(test_y, pred))
        # print(name,
        #       'f1_mi', f1_micro,
        #       'f1_ma', f1_macro,
        #       'f1_wt', f1_weighted,
        #       'f1_bi', f1_binary,
        #       'auc_p', auc_prob,
        #       'auc_l', auc_label,
        #       )
        # sys.stdout.flush()
        return f1_micro, f1_macro, f1_weighted, f1_binary, auc_prob, auc_label, matrix

    def test_mlp(self, reps, test_pos_dir, test_neg_dir, ex=None):
        test_i_index = torch.cat((test_pos_dir[0], test_neg_dir[0]), dim=0).detach().numpy().tolist()
        test_j_index = torch.cat((test_pos_dir[1], test_neg_dir[1]), dim=0).detach().numpy().tolist()
        test_y = np.array([1] * test_pos_dir.shape[1] + [0] * test_neg_dir.shape[1])
        # making directed link sign prediction
        test_x = self.generate_link_representation(reps, test_i_index, test_j_index)
        pred_p = self.mlp(test_x, ex=ex).detach().cpu().numpy()
        pred_p = np.concatenate((1 - pred_p, pred_p), axis=1)
        mlp_dir = self.evaluation(test_y, pred_p, 'mlp_dir')
        # """
        # making undirected link sign prediction
        test_x_inv = self.generate_link_representation(reps, test_j_index, test_i_index)
        pred_p_inv = self.mlp(test_x_inv, ex=ex).detach().cpu().numpy()
        pred_p_inv = np.concatenate((1 - pred_p_inv, pred_p_inv), axis=1)
        #
        pred_p_all = np.concatenate(([pred_p, pred_p_inv]), axis=0)
        test_y_all = np.concatenate((test_y, test_y))
        mlp_dou = self.evaluation(test_y_all, pred_p_all, 'mlp_dou')

        # fold = 4
        # with open('/home/chentao/code/bb/template/runs/[sweep_optuna]seed_4/5_{}/scores.pkl'.format(fold),
        #           'wb') as f:
        #     pickle.dump(pred_p_all, f)
        # with open('/home/chentao/code/bb/template/runs/[sweep_optuna]seed_4/5_{}/labels.pkl'.format(fold),
        #           'wb') as f:
        #     pickle.dump(test_y_all, f)

        return mlp_dir, mlp_dou

    def test_func(self, node_reps, adj_pos, adj_neg, test_pos_dir, test_neg_dir, last_epoch=False, path=None, ex=None):
        node_representations = self.forward(node_reps, adj_pos, adj_neg)

        metrics_mlp_dir, metrics_mlp_dou = self.test_mlp(node_representations, test_pos_dir, test_neg_dir, ex=ex)

        if last_epoch and path is not None:
            if self.cuda:
                node_representations = node_representations.detach().cpu().numpy()
            else:
                node_representations = node_representations.detach().numpy()
            np.savetxt(path, node_representations)

        return metrics_mlp_dir, metrics_mlp_dou


def node_feature_masking(features, mask_ratio=0.1):
    """
    节点特征掩码
    :param features: 节点特征矩阵
    :param mask_ratio: 掩码比例
    :return: 掩码后的节点特征矩阵
    """
    mask = torch.bernoulli(torch.full(features.shape, 1 - mask_ratio)).bool()
    masked_features = features * mask
    return masked_features


def edge_perturbation(adj_matrix, perturb_ratio=0.1):
    """
    边扰动
    :param adj_matrix: 邻接矩阵
    :param perturb_ratio: 扰动比例
    :return: 扰动后的邻接矩阵
    """
    num_edges = int(adj_matrix.sum())
    num_perturb = int(num_edges * perturb_ratio)
    edges = torch.nonzero(adj_matrix)
    idx = np.random.choice(len(edges), num_perturb, replace=False)
    perturbed_edges = edges[idx]

    new_adj_matrix = adj_matrix.clone()
    new_adj_matrix[perturbed_edges[:, 0], perturbed_edges[:, 1]] = 0
    new_adj_matrix[perturbed_edges[:, 1], perturbed_edges[:, 0]] = 0  # 对于无向图

    # 随机添加一些边
    num_nodes = adj_matrix.shape[0]
    new_edges = []
    while len(new_edges) < num_perturb:
        i, j = np.random.randint(0, num_nodes, 2)
        if i != j and new_adj_matrix[i, j] == 0:
            new_adj_matrix[i, j] = 1
            new_adj_matrix[j, i] = 1  # 对于无向图
            new_edges.append((i, j))

    return new_adj_matrix


def subgraph_sampling(adj_matrix, features, sample_ratio=0.8):
    """
    子图采样
    :param adj_matrix: 邻接矩阵
    :param features: 节点特征矩阵
    :param sample_ratio: 采样比例
    :return: 采样后的邻接矩阵和节点特征矩阵
    """
    num_nodes = adj_matrix.shape[0]
    num_sample = int(num_nodes * sample_ratio)
    sampled_nodes = np.random.choice(num_nodes, num_sample, replace=False)
    sampled_adj_matrix = adj_matrix[sampled_nodes][:, sampled_nodes]
    sampled_features = features[sampled_nodes]
    return sampled_adj_matrix, sampled_features

if __name__ == "__main__":
    # import time
    # from torch.optim import Adam
    #
    # # 设置随机种子保证结果可重复
    # torch.manual_seed(0)
    #
    # # 模拟数据生成
    # num_nodes = 1000  # 节点数量
    # num_feats = 64  # 特征维度
    # node_reps = torch.rand(num_nodes, num_feats)  # 随机生成节点特征表示
    # adj_pos = (torch.randint(0, num_nodes, (2, 500)),)  # 随机生成正向链接
    # adj_neg = (torch.randint(0, num_nodes, (2, 500)),)  # 随机生成负向链接
    #
    # # 初始化模型
    # model = SLGNN(num_feats=num_feats)
    # optimizer = Adam(model.parameters(), lr=0.01)
    #
    # # 确保模型和数据都在同一个设备上
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # node_reps = node_reps.to(device)
    #
    # # 前向推理时间测试
    # start_time = time.time()
    # output = model(node_reps, adj_pos, adj_neg)
    # forward_time = time.time() - start_time
    # print(f"前向推理耗时: {forward_time:.6f} 秒")
    #
    # # 设置损失函数和反向传播
    # target = torch.cat([torch.ones(adj_pos[0].shape[1]), torch.zeros(adj_neg[0].shape[1])]).to(device)
    # loss_func = torch.nn.BCELoss()
    #
    # # 计算损失
    # predicted = model.train_prediction(node_reps, adj_pos, adj_neg)
    # predicted = predicted[:, 1]  # 取正类概率
    # loss = loss_func(predicted, target)
    #
    # # 反向传播时间测试
    # start_time = time.time()
    # optimizer.zero_grad()  # 清空梯度
    # loss.backward()  # 计算梯度
    # optimizer.step()  # 更新参数
    # backward_time = time.time() - start_time
    # print(f"反向传播耗时: {backward_time:.6f} 秒")

    def he_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print(m.__name__)


    model = SLGNN(num_feats=64)
    model.apply(he_init)
