import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


def train(model, loss_fn, mining_fn, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        idxs_tuple = mining_fn(embeddings, labels)
        loss = loss_fn(embeddings, labels, idxs_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_fn.num_triplets
                )
            )


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(train_set, test_set, model, accuracy_calculator: AccuracyCalculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)

    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)

    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels)
    print(
        "Test set accuracy (Precision@1) = {}".format(
            accuracies["precision_at_1"])
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

batch_size = 256

train_dataset = datasets.MNIST(
    "data/MNIST", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    "data/MNIST", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
num_epochs = 4

distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_fn = losses.TripletMarginLoss(
    margin=0.2,
    distance=distance,
    reducer=reducer,
)
mining_fn = miners.TripletMarginMiner(
    margin=0.2,
    distance=distance,
    type_of_triplets="semihard",
)
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

if __name__ == "__main__":
    for epoch in range(1, num_epochs + 1):
        train(model, loss_fn, mining_fn, device,
              train_loader, optimizer, epoch)
        test(train_dataset, test_dataset, model, accuracy_calculator)
