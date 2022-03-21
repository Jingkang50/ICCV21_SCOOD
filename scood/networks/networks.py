import torch.nn as nn
import timm


class BaseNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_classes: int = 10,
        dim_aux: int = 0,
    ):
        super().__init__()

        self.backbone = backbone
        self.dim_aux = dim_aux

        self.fc_class = nn.Linear(feat_dim, num_classes)

        if dim_aux > 0:
            self.fc_aux = nn.Linear(feat_dim, dim_aux)
        else:
            self.fc_aux = None

    def forward(self, x, return_feature=False, return_aux=False):
        feats = self.backbone(x)  # (N, feat_dim)
        logits_cls = self.fc_class(feats)

        if return_feature and return_aux:
            logits_aux = self.fc_aux(feats)
            return logits_cls, logits_aux, feats

        elif return_aux:
            logits_aux = self.fc_aux(feats)
            return logits_cls, logits_aux

        elif return_feature:
            return logits_cls, feats

        else:
            return logits_cls


class ResNet50d(BaseNet):
    def __init__(
        self, num_classes: int = 10, dim_aux: int = 0, pretrained: bool = False
    ):
        backbone = timm.create_model("resnet50d", pretrained=pretrained)
        feat_dim = backbone.fc.in_features
        backbone.reset_classifier(0)

        super().__init__(backbone, feat_dim, num_classes=num_classes, dim_aux=dim_aux)


class ResNet18(BaseNet):
    def __init__(
        self,
        num_classes: int = 10,
        dim_aux: int = 0,
        pretrained: bool = False,
        output_stride=8,
    ):
        backbone = timm.create_model(
            "resnet18", output_stride=output_stride, pretrained=pretrained
        )
        feat_dim = backbone.fc.in_features
        backbone.reset_classifier(0)

        super().__init__(backbone, feat_dim, num_classes=num_classes, dim_aux=dim_aux)
