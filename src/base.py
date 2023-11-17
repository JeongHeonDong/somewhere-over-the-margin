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

from src.utils.dataset.cub import CUB

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


# 2. Set the model, optimizer
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
    embedder.parameters(), lr=1e-3, weight_decay=1e-3)

# 3. Set the distance, reducer, loss, sampler, and miner
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)

sampler = None

miner = miners.MultiSimilarityMiner(epsilon=0.1)

# 4. Set the tester
tester = testers.GlobalEmbeddingSpaceTester(
    accuracy_calculator=AccuracyCalculator(include=("precision_at_1",)),
)

# 5. Set the trainer
record_keeper, _, _ = logging_presets.get_record_keeper(
    logging_path, tensorboard_path)
hooks = logging_presets.get_hook_container(
    record_keeper, primary_metric="precision_at_1")
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
    trainer.train(num_epochs=4)
