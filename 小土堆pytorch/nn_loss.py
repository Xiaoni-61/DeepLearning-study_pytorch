import torch
from torch import nn
from torch.nn import L1Loss

inputs = torch.tensor([1, 2, 5], dtype=torch.float32)
targets = torch.tensor([1, 2, 3], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))

targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(inputs, targets)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
print("x" + str(x))

loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)

print(result)
print(result_mse)
print(result_cross)
