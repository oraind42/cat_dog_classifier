import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
from PIL import Image
import torch
from torchvision import models, transforms

from common import CLASSES, CLASS_NAMES_CN
from models import ModelRegistry


PRETRAINED_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_predictor(weight_path="weights/best_resnet18.pth",
                   model_name="resnet18", device="cpu"):
    """
    推理专用：先构建同结构模型，再加载 state_dict。
    model_name 可选: alexnet_manual_scratch / alexnet_manual_pretrained /
                   alexnet_tv / resnet18 / resnet50 / mobilenet_v3_small /
                   shufflenet_v2_x1_0
    """
    device = torch.device(device)
    if model_name in ("resnet18", "resnet50", "mobilenet_v3_small",
                       "shufflenet_v2_x1_0", "alexnet_tv"):
        net = ModelRegistry.build(
            model_name, num_classes=2, pretrained=False, freeze=False
        )
    else:
        net = ModelRegistry.build(
            model_name, num_classes=2, pretrained=False, freeze=False
        )

    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=device, weights_only=True)
        net.load_state_dict(state)
        print(f"[Predictor] 已加载权重 -> {weight_path}")
    else:
        print(f"[Predictor] 警告: 权重文件不存在 {weight_path}，使用随机初始化结果不可信")

    net.to(device)
    net.eval()
    return net, device


@torch.no_grad()
def predict_image(image_source, net, device, transform=None):
    """
    返回 dict: {
        'label': int(0/1), 'class': 'cat'/'dog',
        'confidence': float, 'probabilities': [p_cat, p_dog]
    }
    image_source: 文件路径 or PIL.Image.Image
    """
    if transform is None:
        transform = PRETRAINED_TRANSFORM

    if isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)
    logits = net(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return {
        'label': pred,
        'class': CLASSES[pred],
        'class_cn': CLASS_NAMES_CN[pred],
        'confidence': float(probs[pred]),
        'probabilities': [float(p) for p in probs],
    }


def main():
    parser = argparse.ArgumentParser(description="单张图片推理")
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--weights", default="weights/best_resnet18.pth")
    parser.add_argument("--model", default="resnet18",
                        choices=list(ModelRegistry.list_names()))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    net, device = load_predictor(args.weights, args.model, args.device)
    result = predict_image(args.image, net, device)

    print(f"图片:       {args.image}")
    print(f"预测:       {result['class']} ({result['class_cn']})")
    print(f"置信度:     {result['confidence']:.2%}")
    print(f"各类概率:   猫 {result['probabilities'][0]:.2%} | "
          f"狗 {result['probabilities'][1]:.2%}")


if __name__ == "__main__":
    main()
