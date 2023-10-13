import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./download_data", train=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)






class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.liner1 = Linear(196608,10)

    def forward(self,input):
        output = self.liner1(input)
        return output



tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape)

    output = tudui(output)
    print(output.shape)

