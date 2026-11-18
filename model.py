import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    ResNet50_Weights,
    densenet121,
    resnet50,
)

ARCHS = ("resnet50", "densenet121")


def define_model(arch: str = "resnet50", num_classes: int = 15, pretrained: bool = True) -> nn.Module:
    #V1 weights for both, resnet DEFAULT is V2
    if arch == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {ARCHS}.")
    return model
