from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("data/train/ants_image/0013035.jpg")

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

writer.add_image("totensor", img_tensor)

# Normalize
"""
output[channel] = (input[channel] - mean[channel]) / std[channel]
当我均值标准差都设置为0.5时，那么相当于我的输入为 input[0,1] 那么 output[-1,1]
"""
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)


# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize_tensor = trans_totensor(img_resize)
writer.add_image("img_resize_tensor",img_resize_tensor)

#Compose - resize - 2
trans_resize_2 = transforms.Resize(400) # 短边变为400，长宽比不变
# PIL -> PIL -> tensor  相当于传递参数
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_compose_resize = trans_compose(img)
writer.add_image("comepose_resize",img_compose_resize)

# RandomCrop
tran_random = transforms.RandomCrop(512)
trans_compose = transforms.Compose([tran_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose(img)
    writer.add_image("crop",img_crop,i)


writer.close()
