import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


data_transform = transforms.Compose([
    torchvision.transforms.ToTensor()
])


train_set = torchvision.datasets.CIFAR10(root="./download_data", train=True, transform=data_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./download_data", train=False, transform=data_transform, download=True)

print(test_set[0])
img, target = test_set[0]

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()

