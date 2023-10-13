import torch

import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2 保存参数
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self,x):
        x =self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui.state_dict(), "tudui_method1.pth")