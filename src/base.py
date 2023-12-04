import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import distances, losses, miners, reducers, testers, trainers

from src.utils.dataset.sop import SOP
from src.utils.dataset.cub import CUB

from src.utils.accuracy import CustomAccuracyCalculator

# 0. Set the path, log, and device
trial_name = "base"  # you can change this to whatever you want.
base_dir = f"results/{trial_name}"
os.makedirs(base_dir, exist_ok=True)
logging_path = f"{base_dir}/logs"
tensorboard_path = f"{base_dir}/tensorboard"
model_path = f"{base_dir}/saved_models"

logging.getLogger().setLevel(logging.INFO)
logging.info("Cuda available: {}".format(torch.cuda.is_available()))
writer = SummaryWriter(tensorboard_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Set the dataset, dataloader
dataset = "CUB"
if dataset == "CUB":
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(
            scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4838, 0.5030, 0.4522), (0.1631, 0.1629, 0.1746)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4838, 0.5030, 0.4522), (0.1631, 0.1629, 0.1746)),
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
        transforms.Normalize((0.5794, 0.5388, 0.5044), (0.2183, 0.2218, 0.2225)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5794, 0.5388, 0.5044), (0.2183, 0.2218, 0.2225)),
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
    trunk = torchvision.models.resnet50(pretrained=True)
    trunk_output_size = trunk.fc.in_features
    trunk.fc = nn.Identity()
    trunk.to(device)

    embedder = nn.Sequential(
        nn.Linear(trunk_output_size, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).to(device)

    trunk_optimizer = torch.optim.Adam(
        trunk.parameters(), lr=1e-4, weight_decay=1e-3)
    embedder_optimizer = torch.optim.Adam(
        embedder.parameters(), lr=1e-2, weight_decay=1e-3)

# 3. Set the distance, reducer, loss, sampler, and miner
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)

sampler = None

miner = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")

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

tester = testers.GlobalEmbeddingSpaceTester(
    accuracy_calculator=CustomAccuracyCalculator(include=metrics, k=knn_k),
)

# 5. Set the trainer
record_keeper, _, _ = logging_presets.get_record_keeper(
    logging_path, tensorboard_path)
hooks = logging_presets.get_hook_container(
    record_keeper, primary_metric="recall_at_1")
end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester, {"val": test_dataset}, model_path, 1, 1)
trainer = trainers.MetricLossOnly(
    models={"trunk": trunk, "embedder": embedder},
    batch_size=64,
    sampler=sampler,
    mining_funcs={"tuple_miner": miner},
    loss_funcs={"metric_loss": loss},
    optimizers={
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
    },
    dataset=train_dataset,
    end_of_epoch_hook=end_of_epoch_hook,
)

# 6. Train
if __name__ == "__main__":
    trainer.train(num_epochs=100)
