# (CUB train) mean: [0.4838, 0.5030, 0.4522], std: [0.1631, 0.1629, 0.1746]
# (SOP train) mean: [0.5794, 0.5388, 0.5044], std: [0.2183, 0.2218, 0.2225]
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.utils.dataset.cub import CUB
from src.utils.dataset.sop import SOP


def calc_mean_std(dataset):

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for images, _ in dataloader:
        for image in images:
            mean += image.mean([1, 2])
            std += image.view(3, -1).std(unbiased=False, dim=1)
    mean /= len(dataset)
    std /= len(dataset)
    return mean, std


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = CUB(root="data/CUB_200_2011",
                  mode="train", transform=transform)

    mean, std = calc_mean_std(dataset)
    print(mean, std)

    dataset = SOP(root="data/SOP", mode="train", transform=transform)

    mean, std = calc_mean_std(dataset)
    print(mean, std)
