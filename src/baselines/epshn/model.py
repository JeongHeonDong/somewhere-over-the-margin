import torchvision
import torch.nn as nn


def setModel(model_name, out_dim):
    if model_name == "R18":
        trunk = torchvision.models.resnet18(pretrained=True)
        trunk_output_size = trunk.fc.in_features
        trunk.fc = nn.Identity()

        embedder = nn.Linear(trunk_output_size, out_dim)

        print("Setting model: ResNet18")

    elif model_name == "R50":
        trunk = torchvision.models.resnet50(pretrained=True)
        trunk_output_size = trunk.fc.in_features
        trunk.fc = nn.Identity()

        embedder = nn.Linear(trunk_output_size, out_dim)

        print("Setting model: ResNet50")

    elif model_name == "GBN":
        trunk = torchvision.models.googlenet(
            pretrained=True, transform_input=False)
        trunk.aux_logits = False
        trunk_output_size = trunk.fc.in_features
        trunk.fc = nn.Identity()

        embedder = nn.Linear(trunk_output_size, out_dim)

        print("Setting model: GoogLeNet")
    else:
        print("model is not existed")

    return trunk, embedder
