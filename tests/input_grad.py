import copy

import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.dense1 = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x):
        x = self.dense1(x)
        return x


if __name__ == "__main__":
    sn = SimpleNet(1, 1)
    sn.zero_grad()

    x = torch.ones(1, 1)
    x.requires_grad = True
    print("sn params:")
    for param in sn.parameters():
        print(param.data)


    for _ in range(5):
        y = sn(x)
        y.backward(retain_graph=True)
        print("y: ", y)
        print("df/dx: ", x.grad)
        x = y
        x.retain_grad()

    print("sn params:")
    for param in sn.parameters():
        print(param.data)