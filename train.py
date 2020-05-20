import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import GCN


class Config:
    lr = 0.011
    test_ratio = 1 / 3
    num_classes = 2
    hidden_size = 300
    n_hidden_layer = 2
    num_epochs = 30000


def depart(total, percent):
    choices = np.random.permutation(total)

    selected = choices[0:int(total * percent)]
    unselected = choices[int(total * percent):]
    return selected, unselected


def evaluate(output, labels_e):
    _, labels = output.max(1)
    labels = labels.cpu()
    labels = labels.numpy()

    diff = [1 if i == j else 0 for i, j in zip(labels, labels_e)]

    return sum(diff) / len(labels)


def main():
    from pre_process import preprocess
    feature, a_hat, labels = preprocess()
    print("loaded")

    selected, unselected = depart(len(labels), 1 - Config.test_ratio)
    labels_selected = labels[selected]
    labels_unselected = labels[unselected]

    feature = torch.from_numpy(feature).float().cuda()
    tensor_selected = torch.tensor(labels_selected).long().cuda()
    a_hat = torch.tensor(a_hat).float().cuda()
    net = GCN(a_hat, feature.shape[1], Config.num_classes, Config.hidden_size, Config.n_hidden_layer).cuda()

    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=Config.lr)
    net.train()
    for e in range(Config.num_epochs):
        optimizer.zero_grad()
        output = net(feature)
        loss = criterion(output[selected], tensor_selected)
        loss.backward()
        optimizer.step()

        trained_accuracy = evaluate(output[selected], labels_selected)
        untrained_accuracy = evaluate(output[unselected], labels_unselected)
        print("[Epoch %d]: trained acc: %.7f, untrained acc: %.7f, loss: %.7f" % (
            e, trained_accuracy, untrained_accuracy, loss.detach().cpu().numpy()))


if __name__ == "__main__":
    main()
