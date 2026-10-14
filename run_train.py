import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import json
from pprint import pprint

import torch
from torchvision import transforms

from common import (
    get_loaders, train_and_eval, IMAGENET_MEAN, IMAGENET_STD, get_default_transforms
)
from models import ModelRegistry, init_weight


AUGMENT_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


EXPERIMENTS = {
    "step1_alexnet_scratch": {
        "desc": "第一步：手动 AlexNet 从零训练",
        "model_name": "alexnet_manual_scratch",
        "pretrained": False,
        "freeze": False,
        "augment": False,
        "lr": 1e-4,
        "epochs": 30,
    },
    "step2_alexnet_pretrained": {
        "desc": "第二步：AlexNet 加载 ImageNet 预训练（冻结卷积，只训分类头）",
        "model_name": "alexnet_manual_pretrained",
        "pretrained": True,
        "freeze": True,
        "augment": False,
        "lr": 1e-4,
        "epochs": 15,
    },
    "step3_resnet18_augment": {
        "desc": "第三步：ResNet18 预训练 + 数据增强",
        "model_name": "resnet18",
        "pretrained": True,
        "freeze": True,
        "augment": True,
        "lr": 1e-3,
        "epochs": 10,
    },
    "step4_resnet18_finetune": {
        "desc": "进阶：ResNet18 全模型微调（lr 更小）",
        "model_name": "resnet18",
        "pretrained": True,
        "freeze": False,
        "augment": True,
        "lr": 1e-5,
        "epochs": 10,
    },
}


def run_experiment(name, data_dir="datasets", batch_size=16, device="cpu",
                   save_path=None, extra_epochs=None):
    if name not in EXPERIMENTS:
        raise KeyError(
            f"未知实验 '{name}'，可选: {list(EXPERIMENTS.keys())}"
        )

    cfg = EXPERIMENTS[name]
    if extra_epochs is not None:
        cfg = {**cfg, "epochs": extra_epochs}

    print("=" * 60)
    print(f"▶ {cfg['desc']}")
    print("=" * 60)
    pprint({k: v for k, v in cfg.items()
            if k in ("model_name", "pretrained", "freeze", "augment", "lr", "epochs")})

    train_tf = AUGMENT_TRANSFORM if cfg["augment"] else None
    train_loader, val_loader = get_loaders(
        data_dir=data_dir, batch_size=batch_size, train_transform=train_tf
    )

    net = ModelRegistry.build(
        name=cfg["model_name"],
        num_classes=2,
        pretrained=cfg["pretrained"],
        freeze=cfg["freeze"],
    )

    if cfg["model_name"] == "alexnet_manual_scratch":
        net.apply(init_weight)

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"参数量: trainable={trainable:,}  total={total:,}")

    history = train_and_eval(
        net, train_loader, val_loader,
        epochs=cfg["epochs"], lr=cfg["lr"],
        device=device, save_path=save_path,
    )

    best = max(history, key=lambda h: h['accuracy'])
    print(f"\n✓ 实验 {name} 完成。最佳 Epoch={best['epoch']} Acc={best['accuracy']:.4f}")
    return history, net


def run_model_comparison(data_dir="datasets", batch_size=16, device="cpu",
                         epochs=5, lr=1e-3, weights_dir="weights"):
    """第四步：多个 torchvision 模型横向对比"""
    print("=" * 60)
    print("▶ 第四步：多模型横向对比（ResNet18 / MobileNetV3 / ShuffleNet）")
    print("=" * 60)

    train_loader, val_loader = get_loaders(data_dir=data_dir, batch_size=batch_size)

    os.makedirs(weights_dir, exist_ok=True)
    results = {}
    candidates = ["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"]

    for name in candidates:
        print(f"\n===== {name} =====")
        net = ModelRegistry.build(name, num_classes=2, pretrained=True, freeze=True)
        save_path = os.path.join(weights_dir, f"best_{name}.pth")
        history = train_and_eval(
            net, train_loader, val_loader,
            epochs=epochs, lr=lr, device=device,
            save_path=save_path,
        )
        best_acc = max(h['accuracy'] for h in history)
        results[name] = best_acc

    print("\n===== 横向对比结果 =====")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:24s} 准确率: {acc:.4f}")

    summary_path = os.path.join(weights_dir, "comparison_results.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({n: float(a) for n, a in results.items()}, f, indent=2)
    print(f"\n结果已保存 -> {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="猫狗分类 - 训练入口")
    parser.add_argument(
        "--exp", choices=list(EXPERIMENTS.keys()) + ["compare", "all"],
        default="step3_resnet18_augment",
        help="跑哪个实验（默认 step3_resnet18_augment）"
    )
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖实验默认 epoch 数")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖实验默认学习率")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-path", default=None,
                        help="保存最佳模型的 .pth 路径")
    parser.add_argument("--weights-dir", default="weights")

    args = parser.parse_args()

    if args.exp == "compare":
        run_model_comparison(
            data_dir=args.data_dir, batch_size=args.batch_size,
            device=args.device, epochs=args.epochs or 5,
            weights_dir=args.weights_dir,
        )
    elif args.exp == "all":
        for name in EXPERIMENTS:
            path = os.path.join(args.weights_dir, f"best_{name}.pth")
            run_experiment(name, args.data_dir, args.batch_size, args.device, path)
    else:
        if args.save_path is None:
            args.save_path = os.path.join(args.weights_dir, f"best_{args.exp}.pth")
        run_experiment(
            args.exp, args.data_dir, args.batch_size, args.device,
            args.save_path, extra_epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
