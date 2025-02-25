import torch
import torch.nn as nn
from .layer_tdhnn import SpMergeAttentionLayer

act_func = nn.ReLU()


# act_func = nn.PReLU()


class LayerAggregator(nn.Module):
    def __init__(self, dim_in=64, dim_out=64, num_edges=100, nheads=4, alpha=0.2, node_dropout=0.5, att_dropout=0.5,
                 last_layer=False, bias_hgnn=True, up_bound=0.95, low_bound=0.9,
                 min_num_edges=64, k_n=10, k_e=10):
        super(LayerAggregator, self).__init__()
        self.act_func = act_func
        self.last_layer = last_layer
        self.nheads = nheads

        self.attentions = [
            SpMergeAttentionLayer(dim_in=dim_in, dim_out=dim_out, alpha=alpha, node_dropout=node_dropout,
                                  att_dropout=att_dropout, bias_hgnn=bias_hgnn, up_bound=up_bound, low_bound=low_bound,
                                  min_num_edges=min_num_edges, k_n=k_n, k_e=k_e, num_edges=num_edges) for att_id in
            range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        if last_layer:
            self.layer_norm = nn.LayerNorm(dim_out)
            self.bn = nn.BatchNorm1d(dim_out)
            self.linear = nn.Linear(dim_in, dim_out)
        else:
            self.layer_norm = nn.LayerNorm(dim_out * nheads)
            self.bn = nn.BatchNorm1d(dim_out * nheads)
            self.linear = nn.Linear(dim_in, dim_out * nheads)

    def forward(self, node_reps, adj_pos, adj_neg):
        row_indices_self = [i for i in range(node_reps.size()[0])]
        adj_self = torch.LongTensor([row_indices_self, row_indices_self]).to(device=adj_pos.device)
        adj_pos2 = torch.cat((adj_pos, adj_self), dim=1)

        adj_pos2 = adj_pos2.to(node_reps.device)
        adj_neg = adj_neg.to(node_reps.device)
        node_reps = node_reps[torch.LongTensor(row_indices_self).to(node_reps.device)]

        if len(self.attentions) > 1:
            # print("len(self.attentions)", len(self.attentions))
            if self.last_layer:
                h_hidden = 0
                h_hidden_pos = 0
                h_hidden_neg = 0
                for idx, att_layer in enumerate(self.attentions):
                    res, res_pos, res_neg = att_layer(node_reps, adj_pos2, adj_neg,
                                                      shape=(len(row_indices_self), len(row_indices_self)))
                    # print("res", res)
                    h_hidden += res
                    h_hidden_pos += res_pos
                    h_hidden_neg += res_neg
                #       print("h_hidden", h_hidden)
                h_hidden /= len(self.attentions)
                h_hidden_pos /= len(self.attentions)
                h_hidden_neg /= len(self.attentions)
            else:
                #####标记一下,这里重复调用同一个模块了，我思考一下怎么改

                # 调用一次att()函数并存储结果
                outputs = [att(node_reps, adj_pos2, adj_neg, shape=(len(row_indices_self), len(row_indices_self))) for
                           att in self.attentions]

                # 从存储的结果中选择需要的输出进行拼接
                h_hidden = torch.cat([output[0] for output in outputs], dim=1)  # 拼接第一个输出
                h_hidden_pos = torch.cat([output[1] for output in outputs], dim=1)  # 拼接第二个输出
                h_hidden_neg = torch.cat([output[2] for output in outputs], dim=1)  # 拼接第三个输出

                # h_hidden = torch.cat([
                #     att(node_reps, adj_pos2, adj_neg, shape=(len(row_indices_self), len(row_indices_self)))[0] for att
                #     in
                #     self.attentions], dim=1)
                # h_hidden_pos = torch.cat([
                #     att(node_reps, adj_pos2, adj_neg, shape=(len(row_indices_self), len(row_indices_self)))[1] for att
                #     in self.attentions], dim=1)
                # h_hidden_neg = torch.cat([
                #     att(node_reps, adj_pos2, adj_neg, shape=(len(row_indices_self), len(row_indices_self)))[2] for att
                #     in self.attentions], dim=1)

            # print("h_hidden", h_hidden.shape)
        else:
            h_hidden, h_hidden_pos, h_hidden_neg = self.attentions[0](node_reps, adj_pos2, adj_neg,
                                                                      shape=(
                                                                          len(row_indices_self), len(row_indices_self)))

        h_hidden = self.act_func(h_hidden)
        h_hidden_pos = self.act_func(h_hidden_pos)
        h_hidden_neg = self.act_func(h_hidden_neg)

        # h_hidden = self.bn(h_hidden + self.linear(node_reps))
        # h_hidden = h_hidden + self.linear(node_reps)
        # h_hidden = self.bn(h_hidden)

        return h_hidden, h_hidden_pos, h_hidden_neg
