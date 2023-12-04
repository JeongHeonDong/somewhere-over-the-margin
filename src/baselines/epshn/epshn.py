import os
import logging

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from src.utils.dataset.sop import SOP
from src.utils.dataset.cub import CUB

from src.baselines.epshn.model import setModel
from src.baselines.epshn.loss import EPHNLoss
from src.baselines.epshn.sampler import BalanceSampler_filled, BalanceSampler_sample
from src.baselines.epshn.utils import calc_precision_at_1, calc_recall_at_1

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
dataset = "CUB"  # CUB or SOP
if dataset == "CUB":
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(
            scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4838, 0.5030, 0.4522),
                             (0.1631, 0.1629, 0.1746)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4838, 0.5030, 0.4522),
                             (0.1631, 0.1629, 0.1746)),
    ])
    train_dataset = CUB(root="data/CUB_200_2011",
                        mode="train", transform=train_transform)
    test_dataset = CUB(root="data/CUB_200_2011",
                       mode="eval", transform=test_transform)
else:
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(
            scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5794, 0.5388, 0.5044),
                             (0.2183, 0.2218, 0.2225)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5794, 0.5388, 0.5044),
                             (0.2183, 0.2218, 0.2225)),
    ])
    train_dataset = SOP(root="data/SOP",
                        mode="train", transform=train_transform)
    test_dataset = SOP(root="data/SOP",
                       mode="eval", transform=test_transform)

# 2. Set the model, optimizer
out_dim = 64
model_name = "R18"
lr = 3e-2
num_epochs = 100
n_size = 16
batch_size = 128

trunk, embedder = setModel(model_name, out_dim)
trunk.to(device)
embedder.to(device)

trunk_optimizer = optim.SGD(trunk.parameters(), lr=lr, momentum=0.0)
trunk_scheduler = optim.lr_scheduler.MultiStepLR(
    trunk_optimizer,  [int(num_epochs*0.5), int(num_epochs*0.75)], gamma=0.1)
embedder_optimizer = optim.SGD(
    embedder.parameters(), lr=lr, momentum=0.0, weight_decay=1e-4)
embedder_scheduler = optim.lr_scheduler.MultiStepLR(
    embedder_optimizer,  [int(num_epochs*0.5), int(num_epochs*0.75)], gamma=0.1)


# 3. Set the distance, reducer, loss, sampler, and miner
loss_fn = EPHNLoss()
loss_fn.semi = True


intervals = []
start = -1
cur_cls = -1
for idx, y in enumerate(train_dataset.ys):
    if y != cur_cls:
        if start != -1:
            intervals.append((start, idx - 1))
        start = idx
        cur_cls = y
intervals.append((start, len(train_dataset.ys) - 1))

sampler = BalanceSampler_filled(
    intervals, GSize=n_size) if dataset == "CUB" else BalanceSampler_sample(intervals, GSize=n_size)

miner = None

# 6. Train
if __name__ == "__main__":
    global_it = 0
    test_freq = 5
    for epoch in range(num_epochs+1):
        print('Epoch {}/{} \n '.format(epoch, num_epochs) + '-' * 40)

        # train phase
        if epoch > 0:

            # create dataloader
            dataLoader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler)

            # record loss
            L_data, N_data = 0.0, 0

            # iterate batch
            for data in dataLoader:
                trunk_optimizer.zero_grad()
                embedder_optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    inputs_bt, labels_bt = data  # <FloatTensor> <LongTensor>
                    fvec = trunk(inputs_bt.cuda())
                    fvec = embedder(fvec.cuda())
                    loss, Pos_log, Neg_log, margin = loss_fn(
                        fvec, labels_bt.cuda())
                    loss.backward()
                    trunk_optimizer.step()
                    embedder_optimizer.step()
                writer.add_histogram('Pos_hist', Pos_log, global_it)
                writer.add_histogram('Neg_hist', Neg_log, global_it)
                writer.add_scalar('Margin', margin, global_it)
                global_it += 1

                L_data += loss.item()
                N_data += len(labels_bt)
            print()

            writer.add_scalar('loss', L_data/N_data, epoch)
            # adjust the learning rate
            trunk_scheduler.step()
            embedder_scheduler.step()

        # test phase
        if epoch % test_freq == 0:
            dataloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False)
            trunk.eval()
            embedder.eval()
            Fvecs = []
            for data in dataloader:
                inputs_bt, labels_bt = data
                fvec = trunk(inputs_bt.cuda())
                fvec = embedder(fvec.cuda())
                embeddings = fvec.cpu().detach().numpy()
                Fvecs.append(embeddings)
            Fvecs = np.concatenate(Fvecs, axis=0)
            labels = test_dataset.ys
            p_at_1 = calc_precision_at_1(Fvecs, labels)
            r_at_1 = calc_recall_at_1(Fvecs, labels)
            print('P@1: {:.3f}'.format(p_at_1))
            print('R@1: {:.3f}'.format(r_at_1))
            writer.add_scalar('P@1', p_at_1, epoch)
            writer.add_scalar('R@1', r_at_1, epoch)
