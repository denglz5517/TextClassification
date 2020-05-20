# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:58:01 2019

@author: WT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, X_size, Y_size, A_hat, args, bias=True):  # X_size = num features
        super(GCN, self).__init__()
        self.A_hat = torch.tensor(A_hat, requires_grad=False).float()
        self.weight = nn.parameter.Parameter(torch.FloatTensor(X_size, args.hidden_size_1))
        var = 2. / (self.weight.size(1) + self.weight.size(0))
        self.weight.data.normal_(0, var)
        self.weight2 = nn.parameter.Parameter(torch.FloatTensor(args.hidden_size_1, args.hidden_size_2))
        var2 = 2. / (self.weight2.size(1) + self.weight2.size(0))
        self.weight2.data.normal_(0, var2)
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(args.hidden_size_1))
            self.bias.data.normal_(0, var)
            self.bias2 = nn.parameter.Parameter(torch.FloatTensor(args.hidden_size_2))
            self.bias2.data.normal_(0, var2)
        else:
            self.bias = None
            self.bias2 = None
        self.fc1 = nn.Linear(args.hidden_size_2, args.num_classes)
        self.weight3 = nn.parameter.Parameter(torch.FloatTensor(args.hidden_size_2, args.num_classes))
    def forward(self, X):  ### 2-layer GCN architecture
        X = torch.mm(X, self.weight)  # X_size * hidden_size_1
        if self.bias is not None:
            X = (X + self.bias)
        X = F.relu(torch.mm(self.A_hat, X))  # X_size * hidden_size_1
        X = torch.mm(X, self.weight2)  # X_size * hidden_size_2
        if self.bias2 is not None:
            X = (X + self.bias2)
        X = torch.mm(self.A_hat, X)  # X_size * hidden_size_2
        X = F.leaky_relu(X)
        X = torch.mm(X, self.weight3)
        torch.mm(self.A_hat, X)

       #
       # X = self.fc1(X)  # X_size * num_classes ??

        X = F.softmax(X, 1)
        return X


if __name__ == '__main__':
    import train

    train.start()
