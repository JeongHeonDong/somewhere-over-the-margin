import argparse
import os
import shutil
import logging

import numpy as np
import matplotlib.pyplot as plt
import umap
from cycler import cycler

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import distances, miners, reducers, testers, trainers, samplers

from src.utils.dataset.sop import SOP
from src.utils.dataset.cub import CUB

from src.utils.accuracy import CustomAccuracyCalculator
from src.losses.tripletMarginLoss import TripletMarginLoss

# Preliminary: Get the argument
# e.g. (In slurm) python -m src.base --activation gelu --trial base1
parser = argparse.ArgumentParser(description="Select the option to train model.")
parser.add_argument('--activation', type=str, required=True,
                    help='hard_swish, selu, celu, gelu, silu, mish')
parser.add_argument('--trial', type=str, required=True,
                    help='Set the trial name with anything')
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
batch_size = 128

# 1. Set the dataset, dataloader
dataset = "CUB"  # "CUB", "SOP", "MNIST
if dataset == "CUB":
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(
            scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4707, 0.4601, 0.4549),
                             (0.2767, 0.2760, 0.2850)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4707, 0.4601, 0.4549),
                             (0.2767, 0.2760, 0.2850)),
    ])
    train_dataset = CUB(root="data/CUB_200_2011",
                        mode="train", transform=train_transform)
    test_dataset = CUB(root="data/CUB_200_2011",
                       mode="eval", transform=test_transform)
elif dataset == "SOP":
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(
            scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5807, 0.5396, 0.5044),
                             (0.2901, 0.2974, 0.3095)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5807, 0.5396, 0.5044),
                             (0.2901, 0.2974, 0.3095)),
    ])
    train_dataset = SOP(root="data/SOP",
                        mode="train", transform=train_transform)
    test_dataset = SOP(root="data/SOP",
                       mode="eval", transform=test_transform)
elif dataset == "MNIST":
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="data/MNIST", train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root="data/MNIST", train=False, transform=test_transform, download=True)

# 2. Set the model, optimizer
if dataset == "MNIST":
    trunk = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        nn.Flatten(),
    )
    trunk.to(device)

    embedder = nn.Sequential(
        nn.Linear(9216, 128),
    )
    embedder.to(device)

    trunk_optimizer = optim.Adam(trunk.parameters(), lr=1e-2)
    embedder_optimizer = optim.Adam(embedder.parameters(), lr=1e-2)

else:
    trunk = torchvision.models.resnet18(pretrained=True)
    trunk_output_size = trunk.fc.in_features
    trunk.fc = nn.Identity()
    trunk.to(device)

    embedder = nn.Sequential(
        nn.Linear(trunk_output_size, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).to(device)

    trunk_optimizer = torch.optim.Adam(
        trunk.parameters(), lr=1e-5, weight_decay=1e-4)
    embedder_optimizer = torch.optim.Adam(
        embedder.parameters(), lr=1e-4, weight_decay=1e-4)

# 3. Set the distance, reducer, loss, sampler, and miner
distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
reducer = reducers.MeanReducer()
loss_fn = TripletMarginLoss(
    margin=0.1, distance=distance, reducer=reducer, margin_activation=args.activiation)

sampler = samplers.MPerClassSampler(
    train_dataset.classes, m=4, length_before_new_iter=len(train_dataset)
)

miner = miners.MultiSimilarityMiner(epsilon=0.1)

# 4. Set the tester
if dataset == "MNIST":
    metrics = (
        "recall_at_1",
        "recall_at_2",
        "recall_at_4",
        "recall_at_8",
    )
    knn_k = 8
elif dataset == "CUB":
    metrics = (
        "recall_at_1",
        "recall_at_2",
        "recall_at_4",
        "recall_at_8",
    )
    knn_k = 8
elif dataset == "SOP":
    metrics = (
        "recall_at_1",
        "recall_at_10",
        "recall_at_100",
    )
    knn_k = 100

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
    batch_size=batch_size,
    data_device=device,
    visualizer=umap.UMAP(),
    visualizer_hook=visualizer_hook,
)

test_interval = 1
patience = 3
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
    trainer.train(num_epochs=100)
