import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASSES = {0: "cat", 1: "dog"}
CLASS_NAMES_CN = {0: "猫", 1: "狗"}


class FileNameDataset(Dataset):
    """从文件名解析标签：cat.x.jpg -> 0，dog.x.jpg -> 1"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ])
        if len(self.images) == 0:
            raise RuntimeError(
                f"目录 {image_dir} 里没有图片。请把 cat.*.jpg / dog.*.jpg 放进去。"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        label = 0 if 'cat' in img_name.lower() else 1
        if self.transform:
            image = self.transform(image)
        return image, label


def get_default_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def get_loaders(data_dir="datasets", batch_size=16, train_transform=None,
                val_transform=None, seed=42, val_ratio=0.2):
    """统一划分：默认 80% 训练 / 20% 验证，固定随机种子保证可复现"""
    default_train, default_val = get_default_transforms()
    if train_transform is None:
        train_transform = default_train
    if val_transform is None:
        val_transform = default_val

    train_set = FileNameDataset(image_dir=data_dir, transform=train_transform)
    val_set   = FileNameDataset(image_dir=data_dir, transform=val_transform)

    total = len(train_set)
    val_size = int(val_ratio * total)
    train_size = total - val_size

    g = torch.Generator().manual_seed(seed)
    indices = list(range(total))
    train_idx, val_idx = random_split(indices, [train_size, val_size], generator=g)

    train_loader = DataLoader(Subset(train_set, train_idx.indices),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(val_set, val_idx.indices),
                              batch_size=batch_size, shuffle=False)
    print(f"[DataLoader] 训练 {train_size} 张 / 验证 {val_size} 张 (seed={seed})")
    return train_loader, val_loader


def train_one_epoch(net, train_loader, loss_fn, optimizer, device):
    net.train()
    running_loss = 0.0
    correct = total = 0
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        logits = net(X)
        loss = loss_fn(logits, Y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(1)
        correct += (preds == Y).sum().item()
        total += Y.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(net, val_loader, device):
    net.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for X, Y in val_loader:
        X, Y = X.to(device), Y.to(device)
        logits = net(X)
        total_loss += loss_fn(logits, Y).item() * X.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(Y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    n = len(all_labels)
    metrics = {
        'val_loss': total_loss / n,
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'f1': f1,
        'confusion_matrix': cm,
        'all_preds': all_preds,
        'all_labels': all_labels,
    }
    return metrics


def train_and_eval(net, train_loader, val_loader, epochs=5, lr=1e-3,
                   device="cpu", save_path=None):
    """完整训练循环 + 每轮验证 + 返回 history，可选保存最佳模型"""
    device = torch.device(device)
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=lr
    )

    history = []
    best_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(net, train_loader, loss_fn,
                                                 optimizer, device)
        metrics = evaluate(net, val_loader, device)
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} "
            f"Acc: {metrics['accuracy']:.4f} "
            f"P: {metrics['precision']:.4f} "
            f"R: {metrics['recall']:.4f} "
            f"F1: {metrics['f1']:.4f}"
        )
        record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            **metrics,
        }
        history.append(record)

        if save_path and metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(net.state_dict(), save_path)
            print(f"  ✓ 最佳模型已保存 -> {save_path} (Acc={best_acc:.4f})")

    print(f"\n训练完成。最佳验证准确率: {best_acc:.4f}")
    return history
