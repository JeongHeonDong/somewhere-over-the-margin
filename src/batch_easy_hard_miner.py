import argparse
import os
import random
import shutil
import logging

import numpy as np
import matplotlib.pyplot as plt
import umap
from cycler import cycler

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import distances, miners, reducers, testers, trainers

from src.utils.dataset.cub import CUB

from src.utils.accuracy import CustomAccuracyCalculator
from src.losses.tripletMarginLoss import TripletMarginLoss

# Preliminary: Get the argument
# e.g. (In slurm) python -m src.base --activation gelu --trial base1
parser = argparse.ArgumentParser(description="Select the option to train model.")
parser.add_argument('--activation', type=str, required=True,
                    help='relu, soft_plus, leaky_relu, hard_swish, selu, celu, gelu, silu, mish')
parser.add_argument('--trial', type=str, required=True,
                    help='Set the trial name with anything')
parser.add_argument('--seed', type=int, default=1225, required=False, help='Set the seed')
args = parser.parse_args()
logging.info(args)

# 0. Set the path, log, and device
trial_name = args.trial  # you can change this to whatever you want.
base_dir = f"results/{trial_name}"
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir)
logging_path = f"{base_dir}/logs"
tensorboard_path = f"{base_dir}/tensorboard"
model_path = f"{base_dir}/saved_models"

logging.getLogger().setLevel(logging.INFO)
logging.info("Cuda available: {}".format(torch.cuda.is_available()))
writer = SummaryWriter(tensorboard_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# 1. Set the dataset, dataloader
imgsize = 256
train_transform = transforms.Compose([transforms.Resize(int(imgsize*1.1)),
                                                    transforms.RandomCrop(imgsize),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4707, 0.4601, 0.4549), (0.2767, 0.2760, 0.2850))])
    
test_transform = transforms.Compose([transforms.Resize(imgsize),
                                                transforms.CenterCrop(imgsize),
                                                transforms.ToTensor(),
                                                    transforms.Normalize((0.4707, 0.4601, 0.4549), (0.2767, 0.2760, 0.2850))])
train_dataset = CUB(root="data/CUB_200_2011",
                    mode="train", transform=train_transform)
test_dataset = CUB(root="data/CUB_200_2011",
                    mode="eval", transform=test_transform)

# 2. Set the trunk, embedder, optimizers
trunk = torchvision.models.resnet18(pretrained=True)
trunk_output_size = trunk.fc.in_features
trunk.fc = nn.Identity()
trunk.to(device)

embedder = nn.Sequential(
    nn.Linear(trunk_output_size, 64),
).to(device)

trunk_optimizer = torch.optim.Adam(
    trunk.parameters(), lr=1e-5, weight_decay=1e-4)
embedder_optimizer = torch.optim.Adam(
    embedder.parameters(), lr=1e-4, weight_decay=1e-4)

# 3. Set the distance, reducer, loss, sampler, and miner
distance = distances.CosineSimilarity()
reducer = reducers.MeanReducer()
loss_fn = TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer, margin_activation=args.activation)
sampler = None
miner = miners.BatchEasyHardMiner(allowed_pos_range=(0.2, 1), allowed_neg_range=(0.2, 1), distance=distance)

# 4. Set the tester

metrics = (
    "recall_at_1",
    "recall_at_2",
    "recall_at_4",
    "recall_at_8",
)
knn_k = 8
batch_size = 40

record_keeper, _, _ = logging_presets.get_record_keeper(
    logging_path, tensorboard_path)
hooks = logging_presets.get_hook_container(
    record_keeper, primary_metric="recall_at_1")


def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, epoch):
    logging.info(
        "UMAP plot for the {} split and epoch # {}".format(split_name, epoch))
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(20, 15), frameon=False)
    plt.gca().set_prop_cycle(
        cycler("color", [plt.cm.nipy_spectral(i)
               for i in np.linspace(0, 0.9, num_classes)])
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0],
                 umap_embeddings[idx, 1], ".", markersize=1)
    plt.savefig(f"{base_dir}/{split_name}_{epoch}.png")

tester = testers.GlobalEmbeddingSpaceTester(
    accuracy_calculator=CustomAccuracyCalculator(include=metrics, k=knn_k),
    end_of_testing_hook=hooks.end_of_testing_hook,
    batch_size=32,
    visualizer=umap.UMAP(),
    visualizer_hook=visualizer_hook,
)

test_interval = 1
patience = 5
num_epochs = 200

end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester, {"val": test_dataset}, model_path, test_interval, patience)

# 5. Set the trainer
trainer = trainers.MetricLossOnly(
    models={"trunk": trunk, "embedder": embedder},
    batch_size=batch_size,
    sampler=sampler,
    mining_funcs={"tuple_miner": miner},
    loss_funcs={"metric_loss": loss_fn},
    optimizers={
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
    },
    dataset=train_dataset,
    end_of_iteration_hook=hooks.end_of_iteration_hook,
    end_of_epoch_hook=end_of_epoch_hook,
)

# 6. Train
if __name__ == "__main__":
    trainer.train(num_epochs=num_epochs)
