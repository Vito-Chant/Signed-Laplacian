import argparse
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import StratifiedKFold
import pickle
import warnings

# from imblearn.over_sampling import BorderlineSMOTE
# from scipy.sparse import csr_matrix
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        # a = a.to_sparse_csr()
        ctx.save_for_backward(a, b)
        ctx.n_row = shape[0]

        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.n_row + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    @staticmethod
    def forward(indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_undirected_networks(file_name, undirected=True):
    links = {}
    print(file_name)
    with open(file_name) as fp:
        n, m = [int(val) for val in fp.readline().split()[-2:]]
        for line in fp:
            line = line.strip()
            if line == "" or "#" in line:
                continue
            rater, rated, sign = [int(val) for val in line.split()]
            assert (sign != 0)
            sign = 1 if sign > 0 else -1

            if not undirected:
                edge1 = (rater, rated)
                if edge1 not in links:
                    links[edge1] = sign
                elif links[edge1] == sign:
                    pass
                else:
                    links[edge1] = -1
                continue

            edge1, edge2 = (rater, rated), (rated, rater)
            if edge1 not in links:
                links[edge1], links[edge2] = sign, sign
            elif links[edge1] == sign:
                pass
            else:
                links[edge1], links[edge2] = -1, -1

    adj_lists_pos, adj_lists_neg = defaultdict(set), defaultdict(set)
    num_edges_pos, num_edges_neg = 0, 0
    for (i, j), s in links.items():
        if s > 0:
            adj_lists_pos[i].add(j)
            num_edges_pos += 1
        else:
            adj_lists_neg[i].add(j)
            num_edges_neg += 1
    num_edges_pos /= 2
    num_edges_neg /= 2
    return n, [num_edges_pos, num_edges_neg], adj_lists_pos, adj_lists_neg


def load_sparse_adjacency(file_name, undirected=True):
    n, [num_edges_pos, num_edges_neg], adj_lists_pos, adj_lists_neg = load_undirected_networks(file_name, undirected)
    adj_spr_pos_row, adj_spr_pos_col = [], []
    for i in range(n):
        for j in adj_lists_pos[i]:
            adj_spr_pos_row.append(i)
            adj_spr_pos_col.append(j)
    adj_sps_pos = torch.LongTensor([adj_spr_pos_row, adj_spr_pos_col])

    adj_sps_neg_row, adj_sps_neg_col = [], []
    for i in range(n):
        for j in adj_lists_neg[i]:
            adj_sps_neg_row.append(i)
            adj_sps_neg_col.append(j)
    adj_sps_neg = torch.LongTensor([adj_sps_neg_row, adj_sps_neg_col])

    return n, [num_edges_pos, num_edges_neg], adj_sps_pos, adj_sps_neg


def feature_normalization(feature, _type="standard"):
    if _type == "standard":
        return StandardScaler().fit_transform(feature)
    elif _type == "norm_l2":
        return Normalizer().fit_transform(feature)
    else:
        return feature


def read_in_feature_data(feature_train, features_dims):
    print("loading features ... ")
    feat_data = pickle.load(open(feature_train, "rb"))
    print("load tsvd node representation...")
    if features_dims is not None:
        feat_data = feat_data[:, :features_dims]
    num_nodes, num_feats = feat_data.shape
    feat_data = feature_normalization(feat_data, "standard")
    # print(feat_data.shape)
    return num_feats, feat_data


# def load_data(args, abs_path):
#     net_train, feature_train, net_test, features_dims = \
#         abs_path.slgnn[args.model_dataset]['net_train'].format(args.model_dataset_fold), \
#             abs_path.slgnn[args.model_dataset]['features_train'].format(args.model_dataset_fold), \
#             abs_path.slgnn[args.model_dataset]['net_test'].format(args.model_dataset_fold), args.model_feature_dim
#
#     num_nodes, num_edges, adj_sps_pos, adj_sps_neg = load_sparse_adjacency(net_train, undirected=True)
#     _, _, train_pos_dir, train_neg_dir = load_sparse_adjacency(net_train, undirected=False)
#     _, _, test_pos_dir, test_neg_dir = load_sparse_adjacency(net_test, undirected=False)
#     num_feats, feat_data = read_in_feature_data(feature_train, features_dims)
#
#     # adj_sps_neg = torch.sparse_coo_tensor(adj_sps_neg, torch.ones(adj_sps_neg.size()[1]), size=(num_nodes, num_nodes))
#     # adj_sps_neg = adj_sps_neg.to_dense()
#     # adj_sps_neg_resampled = torch.from_numpy(apply_borderline_smote_to_adjacency(adj_sps_neg.numpy())).to(
#     #     torch.LongTensor)
#
#     return num_nodes, num_edges, adj_sps_pos, adj_sps_neg, train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir, num_feats, feat_data
#
def remove_duplicate_edges(pos_edges, neg_edges):
    """
    去除重复的无向边，确保每条无向边只出现一次。
    :param pos_edges: 正边
    :param neg_edges: 负边
    :return: 去重后的正边和负边
    """
    # 通过将每个边按升序排列来消除无向边的重复
    pos_edges_sorted = torch.stack([torch.min(pos_edges, dim=0)[0], torch.max(pos_edges, dim=0)[0]], dim=0)
    neg_edges_sorted = torch.stack([torch.min(neg_edges, dim=0)[0], torch.max(neg_edges, dim=0)[0]], dim=0)

    # 创建一个set来去重
    pos_edges_set = set([tuple(pos_edges_sorted[:, i].tolist()) for i in range(pos_edges_sorted.size(1))])
    neg_edges_set = set([tuple(neg_edges_sorted[:, i].tolist()) for i in range(neg_edges_sorted.size(1))])

    # 转换回tensor格式
    pos_edges_unique = torch.tensor(list(pos_edges_set), dtype=torch.long).T
    neg_edges_unique = torch.tensor(list(neg_edges_set), dtype=torch.long).T

    return pos_edges_unique, neg_edges_unique


def load_complete_network(train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir):
    """
    根据train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir的数据重建完整的网络。
    """
    all_pos_edges = torch.cat([train_pos_dir, test_pos_dir], dim=1)  # 合并训练集和测试集的正边
    all_neg_edges = torch.cat([train_neg_dir, test_neg_dir], dim=1)  # 合并训练集和测试集的负边
    return all_pos_edges, all_neg_edges


def load_data(args, abs_path, seed=None):
    """
    加载数据函数。
    如果 seed 为 None，则按照预先保存的五折数据加载指定折的训练集和测试集。
    如果 seed 不为 None，则根据 seed 重新随机划分五折交叉验证数据集。

    返回：
        num_nodes: 节点数量
        num_edges: [正边数量, 负边数量]
        adj_sps_pos: 正边的稀疏邻接矩阵
        adj_sps_neg: 负边的稀疏邻接矩阵
        train_pos_dir: 训练集正边的有向邻接列表
        train_neg_dir: 训练集负边的有向邻接列表
        test_pos_dir: 测试集正边的有向邻接列表
        test_neg_dir: 测试集负边的有向邻接列表
        num_feats: 特征维度
        feat_data: 节点特征数据
    """
    if seed is None:
        # 按照预先保存的五折数据加载
        net_train = abs_path.slgnn[args.model_dataset]['net_train'].format(args.model_dataset_fold)
        feature_train = abs_path.slgnn[args.model_dataset]['features_train'].format(args.model_dataset_fold)
        net_test = abs_path.slgnn[args.model_dataset]['net_test'].format(args.model_dataset_fold)
        features_dims = args.model_feature_dim

        num_nodes, num_edges, adj_sps_pos, adj_sps_neg = load_sparse_adjacency(net_train, undirected=True)
        _, _, train_pos_dir, train_neg_dir = load_sparse_adjacency(net_train, undirected=False)
        _, _, test_pos_dir, test_neg_dir = load_sparse_adjacency(net_test, undirected=False)
        num_feats, feat_data = read_in_feature_data(feature_train, features_dims)

        return num_nodes, num_edges, adj_sps_pos, adj_sps_neg, train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir, num_feats, feat_data
    else:
        # 如果 seed 不为 None，重新随机划分五折数据集
        net_train = abs_path.slgnn[args.model_dataset]['net_train'].format(args.model_dataset_fold)
        net_test = abs_path.slgnn[args.model_dataset]['net_test'].format(args.model_dataset_fold)
        feature_train = abs_path.slgnn[args.model_dataset]['features_train'].format(args.model_dataset_fold)

        # 加载已有数据，得到所有的正负边数据
        num_nodes, _, train_pos_dir, train_neg_dir = load_sparse_adjacency(net_train, undirected=False)
        _, _, test_pos_dir, test_neg_dir = load_sparse_adjacency(net_test, undirected=False)

        # 重建完整网络
        all_pos_edges, all_neg_edges = load_complete_network(train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir)

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            # 随机划分五折
            pos_edge_indices = torch.randperm(all_pos_edges.size(1))
            neg_edge_indices = torch.randperm(all_neg_edges.size(1))

        # 按照args.model_dataset_fold进行划分，确保每一折中正负样本的比例一致
        fold_size_pos = all_pos_edges.size(1) // 5
        fold_size_neg = all_neg_edges.size(1) // 5

        # 根据fold编号划分数据集
        test_pos_start = args.model_dataset_fold * fold_size_pos
        test_pos_end = (args.model_dataset_fold + 1) * fold_size_pos
        test_neg_start = args.model_dataset_fold * fold_size_neg
        test_neg_end = (args.model_dataset_fold + 1) * fold_size_neg

        test_pos_dir = all_pos_edges[:, pos_edge_indices[test_pos_start:test_pos_end]]
        test_neg_dir = all_neg_edges[:, neg_edge_indices[test_neg_start:test_neg_end]]

        train_pos_dir = torch.cat([all_pos_edges[:, pos_edge_indices[:test_pos_start]],
                                   all_pos_edges[:, pos_edge_indices[test_pos_end:]]], dim=1)
        train_neg_dir = torch.cat([all_neg_edges[:, neg_edge_indices[:test_neg_start]],
                                   all_neg_edges[:, neg_edge_indices[test_neg_end:]]], dim=1)

        # 生成无向边：将正边和负边反向并合并
        adj_sps_pos = torch.cat([train_pos_dir, train_pos_dir.flip(0)], dim=1)
        adj_sps_neg = torch.cat([train_neg_dir, train_neg_dir.flip(0)], dim=1)
        # print(train_pos_dir.shape, train_neg_dir.shape)

        # 去重：去除重复的无向边
        # adj_sps_pos, adj_sps_neg = remove_duplicate_edges(train_pos_dir, train_neg_dir)

        # 更新num_edges（训练集的正负边数）
        num_edges = [train_pos_dir.size(1), train_neg_dir.size(1)]

        # 加载特征数据
        num_feats, feat_data = read_in_feature_data(feature_train, args.model_feature_dim)

        return num_nodes, num_edges, adj_sps_pos, adj_sps_neg, train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir, num_feats, feat_data


def calculate_class_weights(num_pos, num_neg):
    num_edges = num_pos + num_neg
    w_pos_neg = 1
    w_pos = round(w_pos_neg * num_neg / num_edges, 2)
    w_neg = round(w_pos_neg - w_pos, 2)
    return w_pos, w_neg


# def apply_borderline_smote_to_adjacency(adj_sps_neg):
#     # Convert sparse matrix to dense matrix
#     # adj_dense = adj_sps_neg.toarray()
#
#     # Find indices where there are no edges (negative samples)
#     adj_dense=adj_sps_neg
#     neg_indices = np.where(adj_dense == 0)
#
#     # Prepare data for Borderline SMOTE
#     X_neg = np.array(neg_indices).T  # X_neg should be a 2D array of coordinates
#
#     # Initialize Borderline SMOTE
#     smote = BorderlineSMOTE(random_state=42)
#
#     # Generate synthetic samples for negative edges
#     X_neg_resampled, _ = smote.fit_resample(X_neg, np.zeros(len(X_neg)))
#
#     # Update original adjacency matrix with resampled negative edges
#     for sample in X_neg_resampled:
#         adj_dense[tuple(sample)] = 1  # Assume 1 indicates a negative edge
#
#     # Convert back to sparse matrix
#     adj_sps_neg_resampled = csr_matrix(adj_dense)
#
#     return adj_sps_neg_resampled

if __name__ == '__main__':
    import sys
    import os

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from config import abs_path

    args = argparse.Namespace()
    args.model_dataset = "bitcoinAlpha"
    args.model_dataset_fold = 2
    args.model_feature_dim = None

    num_nodes, num_edges, train_pos, train_neg, train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir, num_feats, input_features = load_data(
        args, abs_path, 42)
    print(torch.randn(1))

    args.model_dataset_fold = 2
    num_nodes1, num_edges1, train_pos1, train_neg1, train_pos_dir1, train_neg_dir1, test_pos_dir1, test_neg_dir1, num_feats1, input_features1 = load_data(
        args, abs_path,43)
    print(torch.randn(1))

    # s1 = set()
    # for i in range(train_pos_dir.size(1)):
    #     s1.add((train_pos_dir[0, i].item(), train_pos_dir[1, i].item()))
    # for i in range(train_neg_dir.size(1)):
    #     s1.add((train_neg_dir[0, i].item(), train_neg_dir[1, i].item()))
    # for i in range(test_pos_dir.size(1)):
    #     s1.add((test_pos_dir[0, i].item(), test_pos_dir[1, i].item()))
    # for i in range(test_neg_dir.size(1)):
    #     s1.add((test_neg_dir[0, i].item(), test_neg_dir[1, i].item()))
    #
    # s2= set()
    # for i in range(train_pos_dir1.size(1)):
    #     s2.add((train_pos_dir1[0, i].item(), train_pos_dir1[1, i].item()))
    # for i in range(train_neg_dir1.size(1)):
    #     s2.add((train_neg_dir1[0, i].item(), train_neg_dir1[1, i].item()))
    # for i in range(test_pos_dir1.size(1)):
    #     s2.add((test_pos_dir1[0, i].item(), test_pos_dir1[1, i].item()))
    # for i in range(test_neg_dir1.size(1)):
    #     s2.add((test_neg_dir1[0, i].item(), test_neg_dir1[1, i].item()))
    #
    # print(s1==s2)

    print(train_pos.shape)
    print(train_neg.shape)
    print(num_nodes)

