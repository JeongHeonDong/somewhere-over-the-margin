import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd

activations = ['relu', 'leaky_relu', 'gelu', 'mish', 'silu',
               'hard_swish',  'celu', 'softplus',  'selu',  'nothing']
activations_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
for activation, linestyle in zip(activations, activations_styles):
    df = pd.read_csv(
        "no_miner/no_miner_seed_7777_" + activation + "/logs/accuracies_normalized_GlobalEmbeddingSpaceTester_level_0_VAL_vs_self.csv")
    plt.plot(df[['epoch', 'recall_at_1_level0']].to_numpy(dtype=np.int32)[:, 0],
             df[['epoch', 'recall_at_1_level0']].to_numpy()[:, 1],
             label=activation, linestyle=linestyle)

# relu_df = pd.read_csv(
#     "no_miner/no_miner_seed_7777_relu/logs/accuracies_normalized_GlobalEmbeddingSpaceTester_level_0_VAL_vs_self.csv")

# print(relu_df[['epoch', 'recall_at_1_level0']].to_numpy(dtype=np.int32)[:, 0])
# print(relu_df[['epoch', 'recall_at_1_level0']].to_numpy()[:, 1])

# x = np.arange(-3, 3, 0.1)
# relu_y = F.relu(torch.tensor(x)).numpy()
# leaky_relu_y = F.leaky_relu(torch.tensor(x)).numpy()
# hard_swish_y = F.hardswish(torch.tensor(x)).numpy()
# silu_y = F.silu(torch.tensor(x)).numpy()
# mish_y = F.mish(torch.tensor(x)).numpy()
# softplus_y = F.softplus(torch.tensor(x)).numpy()
# gelu_y = F.gelu(torch.tensor(x)).numpy()
# selu_y = F.selu(torch.tensor(x)).numpy()
# celu_y = F.celu(torch.tensor(x)).numpy()

# plt.plot(x, relu_y, label="ReLU", linestyle="-")
# plt.plot(x, leaky_relu_y, label="LeakyReLU", linestyle="--")
# plt.plot(x, hard_swish_y, label="HardSwish", linestyle="-.")
# plt.plot(x, silu_y, label="SiLU", linestyle=":")
# plt.plot(x, mish_y, label="Mish", linestyle="-")
# plt.plot(x, softplus_y, label="Softplus", linestyle="--")
# plt.plot(x, gelu_y, label="GELU", linestyle="-.")
# plt.plot(x, celu_y, label="CELU", linestyle=":")
# plt.plot(x, selu_y, label="SELU")

plt.legend()
plt.grid(visible=True)
plt.show()
