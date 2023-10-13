import torch
from torch import nn


class lzlmodule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

module = lzlmodule()
x = torch.tensor(1.0)
output = module(x)
print(output)
