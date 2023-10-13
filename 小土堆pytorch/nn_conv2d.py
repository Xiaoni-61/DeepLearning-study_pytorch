import torch

import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./download_data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class lzl(nn.Module):
    def __init__(self):
        super(lzl, self).__init__()
        # in_channels为彩色通道则为3
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


lzl = lzl()

writer = SummaryWriter("./logs")

step = 0
for data in dataloader:
    img, target = data
    print(img.shape)
    output = lzl(img)
    print(output.shape)

    output = torch.reshape(output, (-1, 3, 30, 30))

    writer.add_images("input", img, step)
    writer.add_images("output", output, step)
    step += 1
