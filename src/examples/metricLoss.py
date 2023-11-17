import io
import os
import logging
from PIL import Image
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import umap
from cycler import cycler
from torchvision import datasets, transforms

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


# 0. Set logging and tensorboard stuff
base_dir = "results/examples/metricLoss"
os.makedirs(base_dir, exist_ok=True)

tensorboard_path = f"{base_dir}/tensorboard"
logging_path = f"{base_dir}/logs"
model_path = f"{base_dir}/saved_models"

logging.getLogger().setLevel(logging.INFO)
logging.info("Cuda available: {}".format(torch.cuda.is_available()))
writer = SummaryWriter(tensorboard_path)


# 1. Simple MLP
class MLP(nn.Module):
    def __init__(self, layer_sizes: List[int], final_relu: bool = False):
        super().__init__()
        layer_list = []
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(num_layers):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)


# 2-1. Initilization (Model, Optimizer, Image Transformations)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2-2. Set trunk(pre-trained) model and replace softmax layer with an identity function
trunk = torchvision.models.resnet18(pretrained=True)
trunk_output_size = trunk.fc.in_features
trunk.fc = nn.Identity()
trunk.to(device)

# 2-3. Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
embedder = MLP([trunk_output_size, 64]).to(device)

# 2-4. Set optimizers
trunk_optimizer = torch.optim.Adam(
    trunk.parameters(), lr=1e-5, weight_decay=1e-4)
embedder_optimizer = torch.optim.Adam(
    embedder.parameters(), lr=1e-3, weight_decay=1e-3)

# 2-5. Set the image transforms for CIFAR 100
train_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomResizedCrop(
            scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4857, 0.4995, 0.4324], std=[
                             0.1606, 0.1600, 0.1717]),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4857, 0.4995, 0.4324], std=[
                             0.1606, 0.1600, 0.1717]),
    ]
)

# 3. Reorganize CIFAR 100 dataset
# - 50 classes for training, 50 classes for testing

# 3-1. Download the original CIFAR 100 dataset
original_train_dataset = datasets.CIFAR100(
    root="data/CIFAR100",
    train=True,
    transform=None,
    download=True,
)
original_val_dataset = datasets.CIFAR100(
    root="data/CIFAR100",
    train=False,
    transform=None,
    download=True,
)


# 3-2. This will be used to create train and val sets that are class-disjoint
class ClassDisjointCIFAR100(torch.utils.data.Dataset):
    def __init__(self, original_train_dataset: torch.utils.data.Dataset, original_val_dataset: torch.utils.data.Dataset, train: bool, transform: transforms.Compose = None):
        rule = (lambda x: x < 50) if train else (lambda x: x >= 50)
        train_filtered_idxs = [
            i for i, target in enumerate(original_train_dataset.targets) if rule(target)
        ]
        val_filtered_idxs = [
            i for i, target in enumerate(original_val_dataset.targets) if rule(target)
        ]
        self.data = np.concatenate(
            [
                original_train_dataset.data[train_filtered_idxs],
                original_val_dataset.data[val_filtered_idxs],
            ],
            axis=0,
        )
        self.targets = np.concatenate(
            [
                np.array(original_train_dataset.targets)[train_filtered_idxs],
                np.array(original_val_dataset.targets)[val_filtered_idxs],
            ],
            axis=0,
        )
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# 3-3. Class Disjoint CIFAR 100
train_dataset = ClassDisjointCIFAR100(
    original_train_dataset,
    original_val_dataset,
    train=True,
    transform=train_transform,
)
val_dataset = ClassDisjointCIFAR100(
    original_train_dataset,
    original_val_dataset,
    train=False,
    transform=val_transform,
)

# 4. Create the loss, miner, sampler, and package them into dictionaries.

# 4-1. Set the loss function
loss_fn = losses.TripletMarginLoss(margin=0.1)

# 4-2. Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=0.1)

# 4-3. Set the dataloader sampler
sampler = samplers.MPerClassSampler(
    train_dataset.targets, m=4, length_before_new_iter=len(train_dataset)
)

# 4-4. Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {
    "trunk_optimizer": trunk_optimizer,
    "embedder_optimizer": embedder_optimizer,
}
loss_funcs = {"metric_loss": loss_fn}
mining_funcs = {"tuple_miner": miner}

# 5. Define training hyperparameters
batch_size = 32
num_epochs = 4

# 6. Create  Tester and Trainer

# 6-1. Setup variables
record_keeper, _, _ = logging_presets.get_record_keeper(
    logging_path, tensorboard_path)
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
model_folder = model_path


# 6-2. Create the tester
def visulaizer_hook(umapper, umap_embeddings, labels, split_name, keyname, epoch):
    logging.info("UMAP plot for the {} split and label set {}".format(
        split_name, keyname))
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color",
            [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idxs = labels == label_set[i]
        plt.plot(
            umap_embeddings[idxs, 0],
            umap_embeddings[idxs, 1],
            ".",
            markersize=1,
        )
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image).squeeze()
    writer.add_image("UMAP", image, epoch)


tester = testers.GlobalEmbeddingSpaceTester(
    end_of_testing_hook=hooks.end_of_testing_hook,
    visualizer=umap.UMAP(),
    visualizer_hook=visulaizer_hook,
    dataloader_num_workers=1,
    accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
)

# 6-3. Create the trainer
end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester,
    dataset_dict,
    model_folder,
    1,
    1,
)

trainer = trainers.MetricLossOnly(
    models,
    optimizers,
    batch_size,
    loss_funcs,
    train_dataset,
    mining_funcs=mining_funcs,
    sampler=sampler,
    dataloader_num_workers=1,
    end_of_iteration_hook=hooks.end_of_iteration_hook,
    end_of_epoch_hook=end_of_epoch_hook,
)

# 7. Start training!
if __name__ == "__main__":
    trainer.train(num_epochs=num_epochs)
