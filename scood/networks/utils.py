from .resnet18 import ResNet18
from .wrn import WideResNet
from .densenet import DenseNet3
from .networks import ResNet50d
import torch


def get_network(
    name: str, num_classes: int, num_clusters: int = 0, checkpoint: str = None, **kwargs
):
    if name == "res18":
        net = ResNet18(num_classes=num_classes, dim_aux=num_clusters)

    elif name == "wrn":
        net = WideResNet(
            depth=28,
            widen_factor=10,
            dropRate=0.0,
            num_classes=num_classes,
            dim_aux=num_clusters,
        )

    elif name == "densenet":
        net = DenseNet3(
            depth=100,
            growth_rate=12,
            reduction=0.5,
            bottleneck=True,
            dropRate=0.0,
            num_classes=num_classes,
            dim_aux=num_clusters,
        )

    elif name == "resnet50d":
        net = ResNet50d(
            num_classes=num_classes,
            dim_aux=num_clusters,
            **kwargs,
        )

    else:
        raise Exception("Unexpected Network Architecture!")

    if checkpoint:
        net.load_state_dict(torch.load(checkpoint), strict=False)
        print("Model Loading Completed!")

    return net
