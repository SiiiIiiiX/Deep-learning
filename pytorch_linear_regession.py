import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# x=torch.linspace(0,20,500)
# k=3
# b=10
# y=k*x+b
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

x=torch.rand(512)
noise = 0.2 * torch.randn(x.size())
k=3
b=10
y=k*x+b+noise
plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()
