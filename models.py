import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import torch.nn as nn
from torchvision import models


class AlexNetScratch(nn.Module):
    """手动搭建的精简 AlexNet（从零训练）"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, X):
        return self.net(X)


class AlexNetManual(nn.Module):
    """与 torchvision.models.alexnet 结构严格对齐，方便加载官方预训练权重"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),   # index 1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),          # index 4
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),   # index 6 <- 要换这里
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _freeze_backbone(net, head_prefix):
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith(head_prefix)


def _replace_head(net, name, num_classes):
    if name in ("resnet18", "resnet50"):
        in_feat = net.fc.in_features
        net.fc = nn.Linear(in_feat, num_classes)
        return net.fc, "fc"
    elif name == "mobilenet_v3_small":
        head = net.classifier
        in_feat = head[3].in_features
        head[3] = nn.Linear(in_feat, num_classes)
        return head[3], "classifier"
    elif name == "shufflenet_v2_x1_0":
        in_feat = net.fc.in_features
        net.fc = nn.Linear(in_feat, num_classes)
        return net.fc, "fc"
    elif name == "alexnet_tv":
        in_feat = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(in_feat, num_classes)
        return net.classifier[6], "classifier"
    else:
        raise ValueError(f"不支持的模型: {name}")


class ModelRegistry:
    """统一注册：name -> (constructor, has_pretrained_weights)"""

    _REGISTER = {}

    @classmethod
    def register(cls, name, constructor, has_weights=True):
        cls._REGISTER[name] = (constructor, has_weights)

    @classmethod
    def list_names(cls):
        return list(cls._REGISTER.keys())

    @classmethod
    def build(cls, name, num_classes=2, pretrained=False, freeze=True):
        """
        pretrained=False 时只建结构；True 时灌入官方预训练权重并换分类头。
        freeze=True 冻结 backbone，只训练分类头（CPU 友好）。
        """
        if name not in cls._REGISTER:
            raise ValueError(
                f"未知模型 '{name}'，可选: {cls.list_names()}"
            )

        constructor, has_weights = cls._REGISTER[name]

        if name == "alexnet_manual_scratch":
            return AlexNetScratch(num_classes=num_classes)

        if name == "alexnet_manual_pretrained":
            net = AlexNetManual(num_classes=1000)
            if pretrained:
                official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
                net.load_state_dict(official.state_dict())
            net.classifier[6] = nn.Linear(4096, num_classes)
            if freeze:
                _freeze_backbone(net, "classifier")
            return net

        # ---- torchvision 系列 ----
        weights = "DEFAULT" if (pretrained and has_weights) else None
        tv_name = name.replace("_tv", "")
        if tv_name == "alexnet":
            net = models.alexnet(weights=weights)
        elif tv_name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights else None)
        elif tv_name == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if weights else None)
        elif tv_name == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT if weights else None
            )
        elif tv_name == "shufflenet_v2_x1_0":
            net = models.shufflenet_v2_x1_0(
                weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT if weights else None
            )
        else:
            raise ValueError(f"不支持的 torchvision 模型: {name}")

        _, head_prefix = _replace_head(net, tv_name, num_classes)
        if freeze:
            _freeze_backbone(net, head_prefix)
        return net


# 注册所有模型
ModelRegistry.register("alexnet_manual_scratch", None, has_weights=False)
ModelRegistry.register("alexnet_manual_pretrained", None, has_weights=True)
ModelRegistry.register("alexnet_tv", None, has_weights=True)
ModelRegistry.register("resnet18", None, has_weights=True)
ModelRegistry.register("resnet50", None, has_weights=True)
ModelRegistry.register("mobilenet_v3_small", None, has_weights=True)
ModelRegistry.register("shufflenet_v2_x1_0", None, has_weights=True)


def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
