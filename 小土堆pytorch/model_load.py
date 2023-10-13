import torch
import torchvision
from torch import nn

model = torch.load("vgg16_method1.pth")

print(model)

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
model = torch.load("vgg16_method2.pth")
print(vgg16)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self,x):
        x =self.conv1(x)
        return x

tudui = Tudui()
tudui.load_state_dict(torch.load("tudui_method1.pth"))

print(tudui)