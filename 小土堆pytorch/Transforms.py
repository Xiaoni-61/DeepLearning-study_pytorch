from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 工具箱
'''
两个问题：
1 transforms该如何使用
2 为什么我们需要Tensor类型
'''

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1 transforms该如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

# 2 为什么我们需要Tensor类型
# 反向传播的属性 梯度

writer.add_image("tensor_img", tensor_img)

writer.close()