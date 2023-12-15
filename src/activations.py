import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import numpy as np


x = np.arange(-3, 3, 0.1)
relu_y = F.relu(torch.tensor(x)).numpy()
leaky_relu_y = F.leaky_relu(torch.tensor(x)).numpy()
hard_swish_y = F.hardswish(torch.tensor(x)).numpy()
silu_y = F.silu(torch.tensor(x)).numpy()
mish_y = F.mish(torch.tensor(x)).numpy()
softplus_y = F.softplus(torch.tensor(x)).numpy()
gelu_y = F.gelu(torch.tensor(x)).numpy()
selu_y = F.selu(torch.tensor(x)).numpy()
celu_y = F.celu(torch.tensor(x)).numpy()

plt.plot(x, relu_y, "ReLU")
plt.plot(x, leaky_relu_y, "LeakyReLU")
plt.plot(x, mish_y, "Mish")
plt.plot(x, hard_swish_y, "HardSwish")
plt.plot(x, silu_y, "SiLU")
plt.plot(x, softplus_y, "Softplus")
plt.plot(x, gelu_y, "GELU")
plt.plot(x, selu_y, "SELU")
plt.plot(x, celu_y, "CELU")
plt.legend()
plt.show()
