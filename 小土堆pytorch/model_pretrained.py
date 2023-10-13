import torchvision

# train_data = torchvision.datasets.ImageNet("./data",split="train",download=True,
#                                            transform=torchvision.transforms.ToTensor())


vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)
pass