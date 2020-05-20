# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:58:01 2019

@author: WT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, a_hat, in_feats, out_feats, bias=True, activation=None):
        super(GraphConvolution, self).__init__()
        self.a_hat = a_hat
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        self.activation = activation

    def forward(self, h):
        h = torch.mm(h, self.weight)
        if self.bias is not None:
            h = (h + self.bias)
        h = torch.mm(self.a_hat, h)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def extra_repr(self):
        shp = self.weight.shape
        summary = 'in={}, out={}'.format(shp[0], shp[1])
        summary += ', bias={}'.format(self.bias is not None)
        summary += ', activation={}'.format(self.activation)
        return summary.format(**self.__dict__)


class GCN(nn.Module):
    def __init__(self, a_hat, in_feats, out_feats, hidden_size, n_hidden_layer, bias=True):  # X_size = num features
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(a_hat, in_feats, hidden_size, activation=F.relu, bias=bias))

        for _ in range(n_hidden_layer):
            self.layers.append(GraphConvolution(a_hat, hidden_size, hidden_size, activation=F.relu, bias=bias))

        self.layers.append(GraphConvolution(a_hat, hidden_size, out_feats, activation=None, bias=bias))

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)

        h = F.softmax(h, dim=1)
        return h


if __name__ == '__main__':
    import train

    train.start()
