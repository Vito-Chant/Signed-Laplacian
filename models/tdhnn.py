import math
import sys
import pickle
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from torch_geometric.nn import GCNConv, GATConv
from .model_m import SLGNN
from sklearn import metrics
import os

act_func = torch.nn.ReLU()


# act_func = torch.nn.PReLU()


#################自注意力机制模块##################
# class SelfAttention(nn.Module):
#     def __init__(self, in_features):
#         super(SelfAttention, self).__init__()
#         self.in_features = in_features
#         self.W = nn.Linear(in_features, in_features)
#         self.V = nn.Linear(in_features, 1)
#
#     def forward(self, x):
#         # 计算注意力权重
#         attn_weights = self.V(torch.tanh(self.W(x))).squeeze(dim=-1)
#         attn_weights = torch.softmax(attn_weights, dim=-1)
#
#         # 使用注意力权重对节点表示进行加权求和
#         weighted_sum = torch.matmul(attn_weights.unsqueeze(1), x).squeeze(1)
#         return weighted_sum


class SelfAttention(nn.Module):
    def __init__(self, feature_dim, heads):
        super(SelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.heads = heads
        self.head_dim = feature_dim // heads

        assert self.head_dim * heads == feature_dim, "Feature dimension must be divisible by heads"

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        N, L, D = x.shape  # N: batch size, L: sequence length, D: feature dimension

        # Linear projections
        query = self.query(x).view(N, L, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(N, L, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(N, L, self.heads, self.head_dim).transpose(1, 2)

        # Attention mechanism
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)

        # Aggregate the context vector
        out = torch.matmul(attention, value)
        out = out.transpose(1, 2).contiguous().view(N, L, self.feature_dim)

        return out


class GNNWithSelfAttention(nn.Module):
    def __init__(self, feature_dim, heads):
        super(GNNWithSelfAttention, self).__init__()
        self.attention = SelfAttention(feature_dim, heads)

    def forward(self, node_features):
        # node_features: [num_nodes, 3, feature_dim]
        # Concatenate the three feature vectors for each node
        # concatenated_features = node_features.view(-1, node_features.size(-1) * 3)

        # Apply self-attention to the concatenated features
        attended_features = self.attention(node_features)

        # Reshape back to the original node structure
        attended_features = attended_features.view(-1, 3, attended_features.size(-1))

        # Combine the attended features for each node into a single feature vector
        # You can use different strategies here, e.g., averaging, summing, or a custom combination
        combined_features = attended_features.mean(dim=1)  # Example: averaging

        return combined_features


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=3, pool_types='avg'):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),  # 将特征图展平为NxC
            nn.Linear(gate_channels, gate_channels // reduction_ratio),  # 降维
            act_func,
            nn.Linear(gate_channels // reduction_ratio, gate_channels)  # 升维
        )
        self.pool_types = pool_types

    def forward(self, x):
        # x的尺寸是NxCxD
        channel_att_sum = None
        if self.pool_types == 'avg':
            # 使用平均池化在特征图的深度维度上进行池化
            avg_pool = F.avg_pool1d(x, x.size(2))  # NxCx1
            channel_att_raw = self.mlp(avg_pool)  # NxC
        elif self.pool_types == 'max':
            # 使用最大池化在特征图的深度维度上进行池化
            max_pool = F.max_pool1d(x, x.size(2))  # NxCx1
            channel_att_raw = self.mlp(max_pool)  # NxC

        # 将不同池化方式的结果累加起来
        if channel_att_sum is None:
            channel_att_sum = channel_att_raw
        else:
            channel_att_sum = channel_att_sum + channel_att_raw

        # 使用sigmoid函数进行压缩，并扩展尺寸以匹配原始输入x
        scale = torch.tanh(channel_att_sum).unsqueeze(2).expand_as(x)
        return torch.sum(x * scale, dim=1)


####################################
class HConstructor(nn.Module):
    def __init__(self, num_edges, f_dim, iters=1, eps=1e-8, hidden_dim=128, up_bound=0.95, low_bound=0.9,
                 min_num_edges=64, k_n=10, k_e=10):
        super().__init__()
        self.num_edges = num_edges
        self.edges = torch.randn(num_edges, f_dim)
        self.iters = iters
        self.eps = eps
        self.scale = f_dim ** -0.5
        self.up_bound = up_bound
        self.low_bound = low_bound
        self.k_n = k_n
        self.k_e = k_e
        self.min_num_edges = min_num_edges
        # self.scale = 1

        self.edges_mu = nn.Parameter(torch.randn(1, f_dim))
        self.edges_logsigma = nn.Parameter(torch.zeros(1, f_dim))
        init.xavier_uniform_(self.edges_logsigma)

        self.to_q = nn.Linear(f_dim, f_dim)
        self.to_k = nn.Linear(f_dim, f_dim)
        self.to_v = nn.Linear(f_dim, f_dim)

        self.gru = nn.GRUCell(f_dim, f_dim)

        hidden_dim = max(f_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(f_dim + f_dim, hidden_dim),
            act_func,
            nn.Linear(hidden_dim, f_dim)
        )

        self.norm_input = nn.LayerNorm(f_dim)
        self.norm_edgs = nn.LayerNorm(f_dim)
        self.norm_pre_ff = nn.LayerNorm(f_dim)

    # def mask_attn(self, attn, k):
    #     indices = torch.topk(attn, k).indices
    #     mask = torch.zeros(attn.shape).bool().to(attn.device)
    #     for i in range(attn.shape[0]):
    #         mask[i][indices[i]] = 1
    #     return attn.mul(mask)
    def mask_attn(self, attn, k):
        _, indices = torch.topk(attn, k, dim=-1)
        mask = torch.zeros_like(attn, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, value=True)
        return attn * mask

    def ajust_edges(self, s_level):
        if self.training:
            return

        if s_level > self.up_bound:
            self.num_edges = self.num_edges + 1
        elif s_level < self.low_bound:
            self.num_edges = self.num_edges - 1
            self.num_edges = max(self.num_edges, self.min_num_edges)
        else:
            return

    def forward(self, inputs):
        n, d, device = *inputs.shape, inputs.device
        n_s = self.num_edges

        # ablation 1
        flag = os.environ['AB'] != '0'
        if flag:
            # if self.edges is None:
            mu = self.edges_mu.expand(n_s, -1)
            sigma = self.edges_logsigma.exp().expand(n_s, -1)
            edges = mu + sigma * torch.randn(mu.shape, device=device)  # 重参数技巧 训练分布
        # 形成高斯分布 均值和方差 采样特征（不可微 网络就优化不了） 采样放在网络外面 网络里面的操作可微即可
        # 正态分布近似 正态分布抽样一个样本作为网络输入 再乘上simga 加mu
        else:
            edges = self.edges.to(device)

        inputs = self.norm_input(inputs)

        k, v = self.to_k(inputs), self.to_v(inputs)
        k = act_func(k)
        v = act_func(v)

        for _ in range(self.iters):
            edges = self.norm_edgs(edges)

            # 求结点相对于边的softmax
            q = self.to_q(edges)
            q = act_func(q)

            # dots = torch.einsum('ni,ij->nj', q, k.T) * self.scale
            # ablation 2
            if os.environ['AB'] == '0' or os.environ['AB'] == '1':
                dots = torch.matmul(edges, inputs.transpose(-2, -1)) * self.scale
            else:
                dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)
            attn = self.mask_attn(attn, self.k_n)  # 这个决定边的特征从哪些结点取

            # 更新超边特征
            # updates = torch.einsum('in,nf->if', attn, v)
            # ablation 2
            if os.environ['AB'] == '0' or os.environ['AB'] == '1':
                updates = torch.matmul(attn, inputs)
            else:
                updates = torch.matmul(attn, v)

            edges = torch.cat((edges, updates), dim=1)
            edges = self.mlp(edges)

            # 按边相对于结点的softmax（更新边之后）
            q = self.to_q(inputs)
            k = self.to_k(edges)
            k = act_func(k)
            v = act_func(v)

            # dots = torch.einsum('ni,ij->nj', q, k.T) * self.scale
            dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_v = dots.softmax(dim=1)
            attn_v = self.mask_attn(attn_v, self.k_e)  # 这个决定一个结点属于多少条边 #选k个最大的其余置为0
            H = attn_v

            # 计算边的饱和度
            cc = H.ceil().abs()
            de = cc.sum(dim=0)
            empty = (de == 0).sum()
            s_level = 1 - empty / n_s

            self.ajust_edges(s_level)

            # print("Num edges is: {}; Satuation level is: {}".format(self.num_edges, s_level))

        self.edges = edges

        return edges, H, dots


class HGNN_conv(nn.Module):
    # 一层卷积模块，模型是由很多层堆叠起来的
    def __init__(self, in_ft, out_ft, num_edges, bias=True, up_bound=0.95, low_bound=0.9,
                 min_num_edges=64, k_n=10, k_e=10):
        super(HGNN_conv, self).__init__()

        self.HConstructor = HConstructor(num_edges, in_ft, up_bound=up_bound, low_bound=low_bound,
                                         min_num_edges=min_num_edges, k_n=k_n, k_e=k_e)

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(in_ft, out_ft))
        self.mlp.append(nn.Linear(out_ft, out_ft))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # 所有module 都是为了forward服务的
        #     print(x.shape)

        edges, H, H_raw = self.HConstructor(x)
        edges = edges.matmul(self.weight)
        # 公式（14）原文
        if self.bias is not None:
            edges = edges + self.bias
        nodes = H.matmul(edges)
        x = self.mlp[0](x) + self.mlp[1](nodes)
        x = x + nodes

        # print("#" * 50)
        return x, H, H_raw


class LinkClassifierMLP(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        hidden1, hidden2, output = int(dim // 2), int(dim // 4), 1  # final_dim * 4 => final_dim * 2 => final_dim => 1
        activation = nn.ReLU()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden1),
            nn.Dropout(dropout),
            activation,
            nn.Linear(hidden1, hidden2),
            nn.Dropout(dropout),
            activation,
            nn.Linear(hidden2, output)
        )

    def forward(self, x):
        res = self.layers(x)
        res = torch.clamp(res, -1e10, 1e10)
        return torch.sigmoid(res)


class HGNN_link_prediction(nn.Module):
    # 分类模型
    def __init__(self, dropout=0.5, in_dim=64, hid_dim=64, num_edges=100, alpha=0.2, node_dropout=0.5,
                 att_dropout=0.5, nheads=4, conv_number=1, transfer=1, up_bound=0.95, low_bound=0.9,
                 min_num_edges=64, k_n=10, k_e=10, self_attention_heads=4, only_x=False):
        super(HGNN_link_prediction, self).__init__()
        self.conv_number = conv_number
        self.dropout = dropout
        self.transfer = transfer
        self.only_x = only_x

        # self.linear_backbone = nn.Linear(in_dim,hid_dim)

        self.slgnn_backbone = SLGNN(num_feats=in_dim, dim_hidden=hid_dim, alpha=alpha,
                                    node_dropout=node_dropout,
                                    att_dropout=att_dropout,
                                    nheads=nheads)

        self.convs = nn.ModuleList()
        self.convs_pos = nn.ModuleList()
        self.convs_neg = nn.ModuleList()
        self.transfers = nn.ModuleList()
        self.transfers_pos = nn.ModuleList()
        self.transfers_neg = nn.ModuleList()

        for i in range(self.conv_number):
            self.convs.append(HGNN_conv(hid_dim, hid_dim, num_edges, up_bound=up_bound, low_bound=low_bound,
                                        min_num_edges=min_num_edges, k_n=k_n, k_e=k_e))
            self.transfers.append(nn.Linear(hid_dim, hid_dim))

            self.convs_pos.append(HGNN_conv(hid_dim, hid_dim, num_edges, up_bound=up_bound, low_bound=low_bound,
                                            min_num_edges=min_num_edges, k_n=k_n, k_e=k_e))
            self.transfers_pos.append(nn.Linear(hid_dim, hid_dim))
            self.convs_neg.append(HGNN_conv(hid_dim, hid_dim, num_edges, up_bound=up_bound, low_bound=low_bound,
                                            min_num_edges=min_num_edges, k_n=k_n, k_e=k_e))
            self.transfers_neg.append(nn.Linear(hid_dim, hid_dim))

        # self.self_attention = GNNWithSelfAttention(feature_dim=hid_dim, heads=self_attention_heads)
        self.channel_attention = ChannelGate(3)

        self.link_rep_dimensions = hid_dim * 4 * self.conv_number
        self.mlp = LinkClassifierMLP(self.link_rep_dimensions)

    def forward(self, node_reps, adj_pos, adj_neg):
        # 模型要看forward
        x, x_pos, x_neg = self.slgnn_backbone(node_reps, adj_pos, adj_neg)

        tmp = []
        tmp_pos = []
        tmp_neg = []
        H = []
        H_raw = []
        H_pos = []
        H_neg = []
        H_raw_pos = []
        H_raw_neg = []
        for i in range(self.conv_number):
            #####这里的self.convs[i]也是重复调用了3次
            x, h, h_raw = self.convs[i](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            if self.transfer == 1:
                x = self.transfers[i](x)
                x = F.relu(x)
            tmp.append(x)
            H.append(h)
            H_raw.append(h_raw)

            ###################################pos
            x_pos, h_pos, h_raw_pos = self.convs_pos[i](x_pos)
            x_pos = F.relu(x_pos)
            x_pos = F.dropout(x_pos, training=self.training)
            if self.transfer == 1:
                x_pos = self.transfers_pos[i](x_pos)
                x_pos = F.relu(x_pos)
            tmp_pos.append(x_pos)
            H_pos.append(h_pos)
            H_raw_pos.append(h_raw_pos)
            ##################################
            ###################################neg
            x_neg, h_neg, h_raw_neg = self.convs_neg[i](x_neg)
            x_neg = F.relu(x_neg)
            x_neg = F.dropout(x_neg, training=self.training)
            if self.transfer == 1:
                x_neg = self.transfers_neg[i](x_neg)
                x_neg = F.relu(x_neg)
            tmp_neg.append(x_neg)
            H_neg.append(h_neg)
            H_raw_neg.append(h_raw_neg)
            ##################################

        x = torch.cat(tmp, dim=1)
        x_neg = torch.cat(tmp_neg, dim=1)
        x_pos = torch.cat(tmp_pos, dim=1)

        return x, H, H_raw, x_pos, H_pos, H_raw_pos, x_neg, H_neg, H_raw_neg

    def train_prediction(self, x_all, x_pos, x_neg, adj_pos, adj_neg):
        # x_all, H, H_raw, x_pos, H_pos, H_raw_pos, x_neg, H_neg, H_raw_neg = self.forward(node_reps, adj_pos, adj_neg)
        if self.only_x:
            x = x_all
        else:
            x = x_all + x_pos - x_neg  # 训练时用于做连接预测的特征x是经过融合的，不可以吗？
            # x = x_pos - x_neg
            # x = self.self_attention(torch.concat((x_all.unsqueeze(1), x_pos.unsqueeze(1), x_neg.unsqueeze(1)), dim=1))
            # x = self.channel_attention(
            #     torch.concat((x_all.unsqueeze(1), x_pos.unsqueeze(1), x_neg.unsqueeze(1)), dim=1))

        train_i_index = torch.cat((adj_pos[0], adj_neg[0]), dim=0)
        train_j_index = torch.cat((adj_pos[1], adj_neg[1]), dim=0)

        link_representations = self.generate_link_representation(x, train_i_index, train_j_index)
        positive_prob = self.mlp(link_representations)
        negative_prob = 1 - positive_prob
        probs = torch.cat((negative_prob, positive_prob), dim=1)
        probs = torch.clamp(probs, 1e-10, 1)
        probs_log = torch.log(probs)

        return probs_log

    @staticmethod
    def generate_link_representation(reps, src, dst):
        src_emb, dst_emb = reps[src], reps[dst]
        link_rep = torch.cat((src_emb, dst_emb, src_emb - dst_emb, src_emb * dst_emb), dim=1)
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

        # print(metrics.confusion_matrix(test_y, pred))
        # print(name,
        #       'f1_mi', f1_micro,
        #       'f1_ma', f1_macro,
        #       'f1_wt', f1_weighted,
        #       'f1_bi', f1_binary,
        #       'auc_p', auc_prob,
        #       'auc_l', auc_label,
        #       )
        sys.stdout.flush()
        return f1_micro, f1_macro, f1_weighted, f1_binary, auc_prob, auc_label

    def test_mlp(self, reps, test_pos_dir, test_neg_dir):
        test_i_index = torch.cat((test_pos_dir[0], test_neg_dir[0]), dim=0).detach().numpy().tolist()
        test_j_index = torch.cat((test_pos_dir[1], test_neg_dir[1]), dim=0).detach().numpy().tolist()
        test_y = np.array([1] * test_pos_dir.shape[1] + [0] * test_neg_dir.shape[1])
        # making directed link sign prediction
        test_x = self.generate_link_representation(reps, test_i_index, test_j_index)
        pred_p = self.mlp(test_x).detach().cpu().numpy()
        pred_p = np.concatenate((1 - pred_p, pred_p), axis=1)
        mlp_dir = self.evaluation(test_y, pred_p, 'mlp_dir')
        # """
        # making undirected link sign prediction
        test_x_inv = self.generate_link_representation(reps, test_j_index, test_i_index)
        pred_p_inv = self.mlp(test_x_inv).detach().cpu().numpy()
        pred_p_inv = np.concatenate((1 - pred_p_inv, pred_p_inv), axis=1)
        #
        pred_p_all = np.concatenate(([pred_p, pred_p_inv]), axis=0)
        test_y_all = np.concatenate((test_y, test_y))
        mlp_dou = self.evaluation(test_y_all, pred_p_all, 'mlp_dou')
        # with open('/home/developers/chentao/code/bb/template/runs/[sweep_optuna]bitcoinAlpha_average_0/12_4/scores.pkl',
        #           'wb') as f:
        #     pickle.dump(pred_p_all, f)
        # with open('/home/developers/chentao/code/bb/template/runs/[sweep_optuna]bitcoinAlpha_average_0/12_4/labels.pkl',
        #           'wb') as f:
        #     pickle.dump(test_y_all, f)
        return mlp_dir, mlp_dou

    def test_func(self, node_reps, adj_pos, adj_neg, test_pos_dir, test_neg_dir, last_epoch=False, path=None):
        node_representations, H, H_raw, node_representations_pos, H_pos, H_raw_pos, node_representations_neg, H_neg, H_raw_neg = self.forward(
            node_reps, adj_pos, adj_neg)
        if self.only_x:
            x = node_representations
        else:
            x = node_representations + node_representations_pos - node_representations_neg
            # x = node_representations_pos - node_representations_neg
            # x = self.self_attention(torch.concat((node_representations.unsqueeze(1), node_representations_pos.unsqueeze(1),
            #                                       node_representations_neg.unsqueeze(1)), dim=1))
            # x = self.channel_attention(
            #     torch.concat((node_representations.unsqueeze(1), node_representations_pos.unsqueeze(1),
            #                   node_representations_neg.unsqueeze(1)), dim=1))

        # 但测试的时候用于链接预测的特征却没经过融合，直接用了第一个特征，训练和测试的流程不一致，我昨天也发现了 会影响结果吗，大概率会，改一下吧
        metrics_mlp_dir, metrics_mlp_dou = self.test_mlp(x, test_pos_dir, test_neg_dir)

        if last_epoch and path is not None:
            if self.cuda:
                node_representations = node_representations.detach().cpu().numpy()
            else:
                node_representations = node_representations.detach().numpy()
            np.savetxt(path, node_representations)

        return metrics_mlp_dir, metrics_mlp_dou
#
# import math
# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# from sklearn import metrics
# from .model_m import SLGNN
#
# ################# 自注意力机制模块 ##################
# class SelfAttention(nn.Module):
#     def __init__(self, feature_dim, heads):
#         super(SelfAttention, self).__init__()
#         self.feature_dim = feature_dim
#         self.heads = heads
#         self.head_dim = feature_dim // heads
#
#         assert self.head_dim * heads == feature_dim, "Feature dimension must be divisible by heads"
#
#         self.query = nn.Linear(feature_dim, feature_dim)
#         self.key = nn.Linear(feature_dim, feature_dim)
#         self.value = nn.Linear(feature_dim, feature_dim)
#
#     def forward(self, x):
#         N, L, D = x.shape  # N: batch size, L: sequence length, D: feature dimension
#
#         # Linear projections
#         query = self.query(x).view(N, L, self.heads, self.head_dim).transpose(1, 2)
#         key = self.key(x).view(N, L, self.heads, self.head_dim).transpose(1, 2)
#         value = self.value(x).view(N, L, self.heads, self.head_dim).transpose(1, 2)
#
#         # Attention mechanism
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         attention = F.softmax(scores, dim=-1)
#
#         # Aggregate the context vector
#         out = torch.matmul(attention, value)
#         out = out.transpose(1, 2).contiguous().view(N, L, self.feature_dim)
#
#         return out
#
#
# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=3, pool_types='avg'):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#         )
#         self.pool_types = pool_types
#
#     def forward(self, x):
#         # x's shape is [N, C, D]
#         if self.pool_types == 'avg':
#             avg_pool = F.avg_pool1d(x, x.size(2))  # [N, C, 1]
#             channel_att_raw = self.mlp(avg_pool)  # [N, C]
#         elif self.pool_types == 'max':
#             max_pool = F.max_pool1d(x, x.size(2))  # [N, C, 1]
#             channel_att_raw = self.mlp(max_pool)  # [N, C]
#         else:
#             raise ValueError("Unsupported pool type")
#
#         # Apply sigmoid and reshape
#         scale = torch.sigmoid(channel_att_raw).unsqueeze(2)  # [N, C, 1]
#         return x * scale.expand_as(x)
#
#
# ####################################
# class HConstructor(nn.Module):
#     def __init__(self, num_edges, f_dim, iters=1, eps=1e-8, hidden_dim=128, up_bound=0.95, low_bound=0.9,
#                  min_num_edges=64, k_n=10, k_e=10):
#         super().__init__()
#         self.num_edges = num_edges
#         self.iters = iters
#         self.eps = eps
#         self.scale = f_dim ** -0.5
#         self.up_bound = up_bound
#         self.low_bound = low_bound
#         self.k_n = k_n
#         self.k_e = k_e
#         self.min_num_edges = min_num_edges
#
#         self.edges_mu = nn.Parameter(torch.randn(1, f_dim))
#         self.edges_logsigma = nn.Parameter(torch.zeros(1, f_dim))
#         nn.init.xavier_uniform_(self.edges_logsigma)
#
#         self.to_q = nn.Linear(f_dim, f_dim)
#         self.to_k = nn.Linear(f_dim, f_dim)
#         self.to_v = nn.Linear(f_dim, f_dim)
#
#         self.gru = nn.GRUCell(f_dim, f_dim)
#
#         hidden_dim = max(f_dim, hidden_dim)
#
#         self.mlp = nn.Sequential(
#             nn.Linear(f_dim + f_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, f_dim)
#         )
#
#         self.norm_input = nn.LayerNorm(f_dim)
#         self.norm_edgs = nn.LayerNorm(f_dim)
#         self.norm_pre_ff = nn.LayerNorm(f_dim)
#
#     def mask_attn(self, attn, k):
#         _, indices = torch.topk(attn, k, dim=-1)
#         mask = torch.zeros_like(attn, dtype=torch.bool)
#         mask.scatter_(dim=-1, index=indices, value=True)
#         return attn * mask
#
#     def adjust_edges(self, s_level):
#         if self.training:
#             return
#
#         if s_level > self.up_bound:
#             self.num_edges += 1
#         elif s_level < self.low_bound:
#             self.num_edges = max(self.num_edges - 1, self.min_num_edges)
#         # Else, do nothing
#
#     def forward(self, inputs):
#         n, d = inputs.shape
#         device = inputs.device
#         n_s = self.num_edges
#
#         # Sample edges using reparameterization trick
#         mu = self.edges_mu.expand(n_s, -1)  # [num_edges, f_dim]
#         sigma = self.edges_logsigma.exp().expand(n_s, -1)  # [num_edges, f_dim]
#         edges = mu + sigma * torch.randn(mu.shape, device=device)  # [num_edges, f_dim]
#
#         inputs = self.norm_input(inputs)
#
#         k, v = F.relu(self.to_k(inputs)), F.relu(self.to_v(inputs))  # [n, f_dim]
#
#         for _ in range(self.iters):
#             edges = self.norm_edgs(edges)
#
#             # Compute attention
#             q = F.relu(self.to_q(edges))  # [num_edges, f_dim]
#             dots = torch.matmul(q, k.transpose(0, 1)) * self.scale  # [num_edges, n]
#             attn = F.softmax(dots, dim=-1) + self.eps
#             attn = self.mask_attn(attn, self.k_n)  # [num_edges, n]
#
#             # Update edges
#             updates = torch.matmul(attn, v)  # [num_edges, f_dim]
#             edges = self.mlp(torch.cat((edges, updates), dim=1))  # [num_edges, f_dim]
#
#             # Compute node attention
#             q_nodes = F.relu(self.to_q(inputs))  # [n, f_dim]
#             dots_nodes = torch.matmul(q_nodes, edges.transpose(0, 1)) * self.scale  # [n, num_edges]
#             attn_nodes = F.softmax(dots_nodes, dim=-1)  # [n, num_edges]
#             attn_nodes = self.mask_attn(attn_nodes, self.k_e)  # [n, num_edges]
#
#             # Compute saturation level
#             saturation = 1 - (attn_nodes.sum(dim=0) == 0).sum().item() / n_s
#             self.adjust_edges(saturation)
#
#         self.edges = edges
#
#         return edges, attn, dots
#
#
# class HGNN_conv(nn.Module):
#     # 一层卷积模块，模型是由很多层堆叠起来的
#     def __init__(self, in_ft, out_ft, num_edges, bias=True, up_bound=0.95, low_bound=0.9,
#                  min_num_edges=64, k_n=10, k_e=10):
#         super(HGNN_conv, self).__init__()
#
#         self.HConstructor = HConstructor(num_edges, in_ft, up_bound=up_bound, low_bound=low_bound,
#                                          min_num_edges=min_num_edges, k_n=k_n, k_e=k_e)
#
#         self.weight = Parameter(torch.Tensor(in_ft, out_ft))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_ft))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#         self.mlp = nn.Sequential(
#             nn.Linear(in_ft, out_ft),
#             nn.ReLU(),
#             nn.Linear(out_ft, out_ft)
#         )
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, x):
#         # 所有module 都是为了forward服务的
#         edges, H, H_raw = self.HConstructor(x)  # edges: [num_edges, f_dim]
#
#         edges = torch.matmul(edges, self.weight)  # [num_edges, out_ft]
#         if self.bias is not None:
#             edges = edges + self.bias  # [num_edges, out_ft]
#
#         nodes = torch.matmul(H.T, edges)  # H: [n, num_edges], edges: [num_edges, out_ft] -> [n, out_ft]
#         x = x + nodes  # Residual connection
#         return x, H, H_raw
#
#
# class LinkClassifierMLP(nn.Module):
#     def __init__(self, dim, dropout=0.0):
#         super().__init__()
#         hidden1, hidden2, output = int(dim // 2), int(dim // 4), 1
#         activation = nn.ReLU()
#         self.layers = nn.Sequential(
#             nn.Linear(dim, hidden1),
#             nn.Dropout(dropout),
#             activation,
#             nn.Linear(hidden1, hidden2),
#             nn.Dropout(dropout),
#             activation,
#             nn.Linear(hidden2, output)
#         )
#
#     def forward(self, x):
#         res = self.layers(x)
#         res = torch.clamp(res, -1e10, 1e10)
#         return torch.sigmoid(res)
#
#
# class HGNN_link_prediction(nn.Module):
#     # 分类模型
#     def __init__(self, dropout=0.5, in_dim=64, hid_dim=64, num_edges=100, alpha=0.2, node_dropout=0.5,
#                  att_dropout=0.5, nheads=4, conv_number=1, transfer=1, up_bound=0.95, low_bound=0.9,
#                  min_num_edges=64, k_n=10, k_e=10, self_attention_heads=4, only_x=False):
#         super(HGNN_link_prediction, self).__init__()
#         self.conv_number = conv_number
#         self.dropout = dropout
#         self.transfer = transfer
#         self.only_x = only_x
#
#         self.slgnn_backbone = SLGNN(num_feats=in_dim, dim_hidden=hid_dim, alpha=alpha,
#                                     node_dropout=node_dropout,
#                                     att_dropout=att_dropout,
#                                     nheads=nheads)
#
#         self.convs = nn.ModuleList()
#         self.convs_pos = nn.ModuleList()
#         self.convs_neg = nn.ModuleList()
#         self.transfers = nn.ModuleList()
#         self.transfers_pos = nn.ModuleList()
#         self.transfers_neg = nn.ModuleList()
#
#         for i in range(self.conv_number):
#             self.convs.append(HGNN_conv(hid_dim, hid_dim, num_edges, up_bound=up_bound, low_bound=low_bound,
#                                         min_num_edges=min_num_edges, k_n=k_n, k_e=k_e))
#             self.transfers.append(nn.Linear(hid_dim, hid_dim))
#
#             self.convs_pos.append(HGNN_conv(hid_dim, hid_dim, num_edges, up_bound=up_bound, low_bound=low_bound,
#                                             min_num_edges=min_num_edges, k_n=k_n, k_e=k_e))
#             self.transfers_pos.append(nn.Linear(hid_dim, hid_dim))
#             self.convs_neg.append(HGNN_conv(hid_dim, hid_dim, num_edges, up_bound=up_bound, low_bound=low_bound,
#                                             min_num_edges=min_num_edges, k_n=k_n, k_e=k_e))
#             self.transfers_neg.append(nn.Linear(hid_dim, hid_dim))
#
#         self.channel_attention = ChannelGate(3)
#
#         self.link_rep_dimensions = hid_dim * 4 * self.conv_number
#         self.mlp = LinkClassifierMLP(self.link_rep_dimensions)
#
#     def forward(self, node_reps, adj_pos, adj_neg):
#         x, x_pos, x_neg = self.slgnn_backbone(node_reps, adj_pos, adj_neg)
#
#         tmp = []
#         tmp_pos = []
#         tmp_neg = []
#         H = []
#         H_raw = []
#         H_pos = []
#         H_raw_pos = []
#         H_neg = []
#         H_raw_neg = []
#         for i in range(self.conv_number):
#             ##### 这里的 self.convs[i] 也是重复调用了3次
#             x, h, h_raw = self.convs[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, training=self.training)
#             if self.transfer == 1:
#                 x = self.transfers[i](x)
#                 x = F.relu(x)
#             tmp.append(x)
#             H.append(h)
#             H_raw.append(h_raw)
#
#             ################################### pos
#             x_pos, h_pos, h_raw_pos = self.convs_pos[i](x_pos)
#             x_pos = F.relu(x_pos)
#             x_pos = F.dropout(x_pos, training=self.training)
#             if self.transfer == 1:
#                 x_pos = self.transfers_pos[i](x_pos)
#                 x_pos = F.relu(x_pos)
#             tmp_pos.append(x_pos)
#             H_pos.append(h_pos)
#             H_raw_pos.append(h_raw_pos)
#             ##################################
#             ################################### neg
#             x_neg, h_neg, h_raw_neg = self.convs_neg[i](x_neg)
#             x_neg = F.relu(x_neg)
#             x_neg = F.dropout(x_neg, training=self.training)
#             if self.transfer == 1:
#                 x_neg = self.transfers_neg[i](x_neg)
#                 x_neg = F.relu(x_neg)
#             tmp_neg.append(x_neg)
#             H_neg.append(h_neg)
#             H_raw_neg.append(h_raw_neg)
#             ##################################
#
#         x = torch.cat(tmp, dim=1)
#         x_neg = torch.cat(tmp_neg, dim=1)
#         x_pos = torch.cat(tmp_pos, dim=1)
#
#         return x, H, H_raw, x_pos, H_pos, H_raw_pos, x_neg, H_neg, H_raw_neg
#
#     def train_prediction(self, x_all, x_pos, x_neg, adj_pos, adj_neg):
#         if self.only_x:
#             x = x_all
#         else:
#             x = x_all + x_pos - x_neg  # 融合特征
#
#         train_i_index = torch.cat((adj_pos[0], adj_neg[0]), dim=0)
#         train_j_index = torch.cat((adj_pos[1], adj_neg[1]), dim=0)
#
#         link_representations = self.generate_link_representation(x, train_i_index, train_j_index)
#         positive_prob = self.mlp(link_representations)
#         negative_prob = 1 - positive_prob
#         probs = torch.cat((negative_prob, positive_prob), dim=1)
#         probs = torch.clamp(probs, 1e-10, 1)
#         probs_log = torch.log(probs)
#
#         return probs_log
#
#     @staticmethod
#     def generate_link_representation(reps, src, dst):
#         src_emb, dst_emb = reps[src], reps[dst]
#         link_rep = torch.cat((src_emb, dst_emb, src_emb - dst_emb, src_emb * dst_emb), dim=1)
#         return link_rep
#
#     @staticmethod
#     def evaluation(test_y, pred_p, name=""):
#         pred = np.argmax(pred_p, axis=1)
#         test_y = np.array(test_y)
#         f1_macro = metrics.f1_score(test_y, pred, average='macro')
#         f1_micro = metrics.f1_score(test_y, pred, average='micro')
#         f1_weighted = metrics.f1_score(test_y, pred, average='weighted')
#         f1_binary = metrics.f1_score(test_y, pred, average='binary')
#         auc_prob = metrics.roc_auc_score(test_y, pred_p[:, 1])
#         auc_label = metrics.roc_auc_score(test_y, pred)
#         matrix = metrics.confusion_matrix(test_y, pred)
#
#         return f1_micro, f1_macro, f1_weighted, f1_binary, auc_prob, auc_label, matrix
#
#     def test_mlp(self, reps, test_pos_dir, test_neg_dir):
#         test_i_index = torch.cat((test_pos_dir[0], test_neg_dir[0]), dim=0).detach().cpu().tolist()
#         test_j_index = torch.cat((test_pos_dir[1], test_neg_dir[1]), dim=0).detach().cpu().tolist()
#         test_y = np.array([1] * test_pos_dir.shape[1] + [0] * test_neg_dir.shape[1])
#         # Directed link prediction
#         test_x = self.generate_link_representation(reps, test_i_index, test_j_index)
#         pred_p = self.mlp(test_x).detach().cpu().numpy()
#         pred_p = np.concatenate((1 - pred_p, pred_p), axis=1)
#         mlp_dir = self.evaluation(test_y, pred_p, 'mlp_dir')
#
#         # Undirected link prediction
#         test_x_inv = self.generate_link_representation(reps, test_j_index, test_i_index)
#         pred_p_inv = self.mlp(test_x_inv).detach().cpu().numpy()
#         pred_p_inv = np.concatenate((1 - pred_p_inv, pred_p_inv), axis=1)
#
#         pred_p_all = np.concatenate((pred_p, pred_p_inv), axis=0)
#         test_y_all = np.concatenate((test_y, test_y))
#         mlp_dou = self.evaluation(test_y_all, pred_p_all, 'mlp_dou')
#
#         return mlp_dir, mlp_dou
#
#     def test_func(self, node_reps, adj_pos, adj_neg, test_pos_dir, test_neg_dir, last_epoch=False, path=None):
#         node_representations, H, H_raw, node_representations_pos, H_pos, H_raw_pos, node_representations_neg, H_neg, H_raw_neg = self.forward(
#             node_reps, adj_pos, adj_neg)
#         if self.only_x:
#             x = node_representations
#         else:
#             x = node_representations + node_representations_pos - node_representations_neg
#
#         metrics_mlp_dir, metrics_mlp_dou = self.test_mlp(x, test_pos_dir, test_neg_dir)
#
#         if last_epoch and path is not None:
#             node_representations = node_representations.detach().cpu().numpy()
#             np.savetxt(path, node_representations)
#
#         return metrics_mlp_dir, metrics_mlp_dou
