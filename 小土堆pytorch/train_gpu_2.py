import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
import time

device = torch.device("cuda:1")
train_data = torchvision.datasets.CIFAR10("./download_data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./download_data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(f"训练长度{train_data_size}")
print(f"测试长度{test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

tudui = model.Tudui()
tudui = tudui.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 1e-2
opti = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

total_train_step = 0

total_test_step = 0
epoch = 5

# 添加tensorboard
writer = SummaryWriter("./log_train")

start_time = time.time()

for i in range(epoch):
    print(f"第{i + 1}轮训练开始")

    # 训练开始
    tudui.train()
    for data in train_dataloader:
        img, targets = data
        img =img.to(device)
        targets = targets.to(device)
        output = tudui(img)
        loss = loss_fn(output, targets)

        opti.zero_grad()
        loss.backward()
        opti.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print(f"训练次数:{total_train_step},loss:{loss}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = tudui(imgs)
            loss = loss_fn(output, targets)  # 先output 再targets
            total_test_loss += loss
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"-----整体数据集上的loss：{total_test_loss}-----")
    print(f"-----整体的正确率：{total_accuracy / test_data_size}-----")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_train_step)
    total_test_step += 1

    torch.save(tudui, f"tudui_{i}.pth")
    print("模型已经保存")

writer.close()
