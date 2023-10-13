import torch
import torchvision.transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import model
writer = SummaryWriter("./lzl")


device = torch.device("cuda")
image_path = "./img/img.png"
image = Image.open(image_path)

print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
writer.add_image("l",image)
print(image.shape)

model1 = torch.load("tudui_4.pth")
print(model1)

image = torch.reshape(image, (-1, 3, 32, 32))  # 这个-1 代表batch_size
print("imagesize="+str(image.shape))
'''
采用GPU训练的模型，不能直接在CPU上使用，按照上一个频讲的把图片传入GPU就可以了
'''
image = image.to(device)
model1.eval()
with torch.no_grad():
    output = model1(image)
print(output)
print(output.argmax(1))

writer.close()