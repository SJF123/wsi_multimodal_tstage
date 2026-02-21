'''
import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# 复用你的 ResidualMLP（你项目里已经有）
from src.models.residual_mlp import ResidualMLP


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_class_weights_from_train(splits_csv: str, num_classes: int, device: str):
    df = pd.read_csv(splits_csv)
    train_df = df[df["split"] == "train"]
    counts = train_df["label"].value_counts().sort_index()

    full_counts = np.zeros(num_classes, dtype=np.int64)
    for k, v in counts.items():
        if 0 <= int(k) < num_classes:
            full_counts[int(k)] = int(v)

    print("[Train label counts]", {i: int(full_counts[i]) for i in range(num_classes)})

    w = 1.0 / (full_counts.astype(np.float32) + 1e-6)
    w = w / w.mean()
    class_weights = torch.tensor(w, dtype=torch.float32).to(device)
    print("[Class weights]", class_weights.detach().cpu().numpy())
    return class_weights


# -----------------------------
# Dataset: patch-level
# -----------------------------
class PatchAsSampleDataset(Dataset):
    """
    每个 slide 有 [200,384] patch feats
    我们把 slide 展开成：200 个样本
    每个样本就是一个 patch: [384]，label = slide label
    """
    def __init__(self, splits_csv: str, feat_dir: str, split: str, max_patches: int = 200):
        super().__init__()
        self.feat_dir = feat_dir
        self.max_patches = max_patches

        df = pd.read_csv(splits_csv)
        df = df[df["split"] == split].copy()

        # 你 splits_labeled.csv 里一般会有 slide_id / slide 或 file_stem
        # 我们尽量兼容：优先找 slide_id，再找 slide，再找 file_stem
        if "slide_id" in df.columns:
            slide_col = "slide_id"
        elif "slide" in df.columns:
            slide_col = "slide"
        elif "file_stem" in df.columns:
            slide_col = "file_stem"
        else:
            raise ValueError("splits_csv must have one of columns: slide_id / slide / file_stem")

        self.rows = []
        for _, r in df.iterrows():
            slide_id = str(r[slide_col])
            label = int(r["label"])
            # pt 文件名通常是 slide_id.pt
            pt_path = os.path.join(self.feat_dir, slide_id + ".pt")
            if os.path.exists(pt_path):
                self.rows.append((slide_id, pt_path, label))

        if len(self.rows) == 0:
            raise RuntimeError(f"No slides found for split={split}. Check feat_dir and splits_csv columns.")

        # 建立 “全局 index → (slide_idx, patch_idx)” 映射
        self.index = []
        for si, (_, _, _) in enumerate(self.rows):
            for pi in range(self.max_patches):
                self.index.append((si, pi))

        print(f"[PatchAsSampleDataset] split={split} slides={len(self.rows)} patches(samples)={len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, pi = self.index[idx]
        _, pt_path, label = self.rows[si]

        feats = torch.load(pt_path, map_location="cpu")  # Tensor [200,384]
        # 安全裁剪
        feats = feats[: self.max_patches]  # [max_patches,384]
        x = feats[pi]  # [384]
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# -----------------------------
# Model: Image MLP
# -----------------------------
class ImageMLPClassifier(nn.Module):
    def __init__(self, img_dim=384, hidden=512, num_classes=4, alpha=0.2, learnable_alpha=False, dropout=0.1):
        super().__init__()
        self.img_ft = ResidualMLP(img_dim, hidden=hidden, dropout=dropout, alpha=alpha, learnable_alpha=learnable_alpha)
        self.head = nn.Linear(img_dim, num_classes)

    def forward(self, x):
        # x: [B,384]
        f = self.img_ft(x)        # [B,384]
        logits = self.head(f)     # [B,C]
        return logits, f


@torch.no_grad()
def eval_cls(model, loader, device, title=""):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits.argmax(dim=1)
        ys.extend(y.cpu().tolist())
        ps.extend(pred.cpu().tolist())

    acc = accuracy_score(ys, ps)
    macro_f1 = f1_score(ys, ps, average="macro")

    print(f"\n===== {title} =====")
    print(f"acc={acc:.4f}  macroF1={macro_f1:.4f}")
    print("\n[Classification Report]")
    print(classification_report(ys, ps, digits=4))
    print("\n[Confusion Matrix]\n", confusion_matrix(ys, ps))
    return acc, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", default="data/splits_labeled.csv")
    ap.add_argument("--feat_dir", default="features/vit_patch_feats")
    ap.add_argument("--max_patches", type=int, default=200)
    ap.add_argument("--num_classes", type=int, default=4)

    # model
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--alpha_mode", choices=["fixed", "learnable"], default="fixed")
    ap.add_argument("--dropout", type=float, default=0.1)

    # train
    ap.add_argument("--batch_size", type=int, default=256)  # patch-level 样本多，batch 可以大一点
    ap.add_argument("--epochs", type=int, default=5)        # 先跑5个 epoch 看效果
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    # loss
    ap.add_argument("--use_class_weight", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    # save
    ap.add_argument("--save_img_mlp", default="checkpoints/img_mlp_patch_pretrain.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # dataset
    train_ds = PatchAsSampleDataset(args.splits_csv, args.feat_dir, split="train", max_patches=args.max_patches)
    val_ds   = PatchAsSampleDataset(args.splits_csv, args.feat_dir, split="val",   max_patches=args.max_patches)
    test_ds  = PatchAsSampleDataset(args.splits_csv, args.feat_dir, split="test",  max_patches=args.max_patches)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    learnable_alpha = (args.alpha_mode == "learnable")
    model = ImageMLPClassifier(
        img_dim=384, hidden=512, num_classes=args.num_classes,
        alpha=args.alpha, learnable_alpha=learnable_alpha, dropout=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_class_weight:
        cw = compute_class_weights_from_train(args.splits_csv, args.num_classes, device)
        ce = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    else:
        ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    ensure_dir(os.path.dirname(args.save_img_mlp))

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            total += loss.item()

        # 每个 epoch 做一次 val
        val_acc, val_f1 = eval_cls(model, val_loader, device, title=f"VAL epoch={epoch:02d}")
        print(f"Epoch {epoch:02d} | train_loss={total/len(train_loader):.4f} | val_acc={val_acc:.4f} | val_macroF1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            torch.save({
                "epoch": best_epoch,
                "best_val_macroF1": best_val_f1,
                "state_dict": best_state,
                "args": vars(args),
            }, args.save_img_mlp)

            print(f"[Saved best] {args.save_img_mlp} (epoch={best_epoch}, val_macroF1={best_val_f1:.4f})")

    # test best
    ckpt = torch.load(args.save_img_mlp, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    _ = eval_cls(model, test_loader, device, title="TEST (best checkpoint)")
    print("\nDone. Best epoch:", ckpt["epoch"], "Best val macroF1:", ckpt["best_val_macroF1"])
    print("Saved:", args.save_img_mlp)


if __name__ == "__main__":
    main()
# python -m src.scripts.train_image_mlp_patch_pretrain.py
# 添加可视化模块
'''
import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('Agg')  # 非交互式后端，适合服务器环境

# 复用你的 ResidualMLP（你项目里已经有）
from src.models.residual_mlp import ResidualMLP


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_class_weights_from_train(splits_csv: str, num_classes: int, device: str):
    df = pd.read_csv(splits_csv)
    train_df = df[df["split"] == "train"]
    counts = train_df["label"].value_counts().sort_index()

    full_counts = np.zeros(num_classes, dtype=np.int64)
    for k, v in counts.items():
        if 0 <= int(k) < num_classes:
            full_counts[int(k)] = int(v)

    print("[Train label counts]", {i: int(full_counts[i]) for i in range(num_classes)})

    w = 1.0 / (full_counts.astype(np.float32) + 1e-6)
    w = w / w.mean()
    class_weights = torch.tensor(w, dtype=torch.float32).to(device)
    print("[Class weights]", class_weights.detach().cpu().numpy())
    return class_weights


# -----------------------------
# Dataset: patch-level
# -----------------------------
class PatchAsSampleDataset(Dataset):
    """
    每个 slide 有 [200,384] patch feats
    我们把 slide 展开成：200 个样本
    每个样本就是一个 patch: [384]，label = slide label
    """

    def __init__(self, splits_csv: str, feat_dir: str, split: str, max_patches: int = 200):
        super().__init__()
        self.feat_dir = feat_dir
        self.max_patches = max_patches

        df = pd.read_csv(splits_csv)
        df = df[df["split"] == split].copy()

        # 你 splits_labeled.csv 里一般会有 slide_id / slide 或 file_stem
        # 我们尽量兼容：优先找 slide_id，再找 slide，再找 file_stem
        if "slide_id" in df.columns:
            slide_col = "slide_id"
        elif "slide" in df.columns:
            slide_col = "slide"
        elif "file_stem" in df.columns:
            slide_col = "file_stem"
        else:
            raise ValueError("splits_csv must have one of columns: slide_id / slide / file_stem")

        self.rows = []
        for _, r in df.iterrows():
            slide_id = str(r[slide_col])
            label = int(r["label"])
            # pt 文件名通常是 slide_id.pt
            pt_path = os.path.join(self.feat_dir, slide_id + ".pt")
            if os.path.exists(pt_path):
                self.rows.append((slide_id, pt_path, label))

        if len(self.rows) == 0:
            raise RuntimeError(f"No slides found for split={split}. Check feat_dir and splits_csv columns.")

        # 建立 "全局 index → (slide_idx, patch_idx)" 映射
        self.index = []
        for si, (_, _, _) in enumerate(self.rows):
            for pi in range(self.max_patches):
                self.index.append((si, pi))

        print(f"[PatchAsSampleDataset] split={split} slides={len(self.rows)} patches(samples)={len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, pi = self.index[idx]
        _, pt_path, label = self.rows[si]

        feats = torch.load(pt_path, map_location="cpu")  # Tensor [200,384]
        # 安全裁剪
        feats = feats[: self.max_patches]  # [max_patches,384]
        x = feats[pi]  # [384]
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# -----------------------------
# Model: Image MLP
# -----------------------------
class ImageMLPClassifier(nn.Module):
    def __init__(self, img_dim=384, hidden=512, num_classes=4, alpha=0.2, learnable_alpha=False, dropout=0.1):
        super().__init__()
        self.img_ft = ResidualMLP(img_dim, hidden=hidden, dropout=dropout, alpha=alpha, learnable_alpha=learnable_alpha)
        self.head = nn.Linear(img_dim, num_classes)

    def forward(self, x):
        # x: [B,384]
        f = self.img_ft(x)  # [B,384]
        logits = self.head(f)  # [B,C]
        return logits, f


# -----------------------------
# 新增：可视化函数
# -----------------------------
def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None, class_names=None):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(8, 6))

    # 使用seaborn的热力图绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)))

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(train_losses, train_accs, val_accs, train_f1s, val_f1s, save_dir="plots"):
    """绘制训练曲线图"""
    # 确保保存目录存在
    ensure_dir(save_dir)

    # 定义SCI论文配色
    color_train = '#B3D49B'  # RGB 179 212 155
    color_val = '#CDA3CB'  # RGB 205 163 203

    epochs = range(1, len(train_losses) + 1)

    # 创建图形和子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 训练损失曲线
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=10)

    # 2. 准确率曲线（训练集和验证集）
    axes[1].plot(epochs, train_accs, '-', color=color_train, linewidth=2, label='Train')
    axes[1].plot(epochs, val_accs, '-', color=color_val, linewidth=2, label='Val')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].tick_params(axis='both', which='major', labelsize=10)

    # 3. F1分数曲线（训练集和验证集）
    axes[2].plot(epochs, train_f1s, '-', color=color_train, linewidth=2, label='Train')
    axes[2].plot(epochs, val_f1s, '-', color=color_val, linewidth=2, label='Val')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Macro F1 Score', fontsize=12)
    axes[2].set_title('Training vs Validation F1 Score', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=11)
    axes[2].tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Visualization] Training curves saved to: {save_path}")


def evaluate_and_record(model, loader, device, title=""):
    """评估模型并返回acc和f1分数，同时打印报告（可选）"""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits.argmax(dim=1)
        ys.extend(y.cpu().tolist())
        ps.extend(pred.cpu().tolist())

    acc = accuracy_score(ys, ps)
    macro_f1 = f1_score(ys, ps, average="macro")

    return acc, macro_f1, ys, ps


@torch.no_grad()
def eval_cls(model, loader, device, title="", verbose=True):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits.argmax(dim=1)
        ys.extend(y.cpu().tolist())
        ps.extend(pred.cpu().tolist())

    acc = accuracy_score(ys, ps)
    macro_f1 = f1_score(ys, ps, average="macro")

    if verbose:
        print(f"\n===== {title} =====")
        print(f"acc={acc:.4f}  macroF1={macro_f1:.4f}")
        print("\n[Classification Report]")
        print(classification_report(ys, ps, digits=4))
        print("\n[Confusion Matrix]\n", confusion_matrix(ys, ps))

    return acc, macro_f1, confusion_matrix(ys, ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", default="data/splits_labeled.csv")
    ap.add_argument("--feat_dir", default="features/vit_patch_feats")
    ap.add_argument("--max_patches", type=int, default=200)
    ap.add_argument("--num_classes", type=int, default=4)

    # model
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--alpha_mode", choices=["fixed", "learnable"], default="fixed")
    ap.add_argument("--dropout", type=float, default=0.1)

    # train
    ap.add_argument("--batch_size", type=int, default=256)  # patch-level 样本多，batch 可以大一点
    ap.add_argument("--epochs", type=int, default=5)  # 先跑5个 epoch 看效果
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    # loss
    ap.add_argument("--use_class_weight", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    # 新增：可视化参数
    ap.add_argument("--save_plots", action="store_true", default=True,
                    help="Save training plots and confusion matrices")
    ap.add_argument("--plots_dir", default="plots/image_mlp", help="Directory to save plots")

    # save
    ap.add_argument("--save_img_mlp", default="checkpoints/img_mlp_patch_pretrain.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 确保可视化目录存在
    if args.save_plots:
        ensure_dir(args.plots_dir)

    # dataset
    train_ds = PatchAsSampleDataset(args.splits_csv, args.feat_dir, split="train", max_patches=args.max_patches)
    val_ds = PatchAsSampleDataset(args.splits_csv, args.feat_dir, split="val", max_patches=args.max_patches)
    test_ds = PatchAsSampleDataset(args.splits_csv, args.feat_dir, split="test", max_patches=args.max_patches)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    learnable_alpha = (args.alpha_mode == "learnable")
    model = ImageMLPClassifier(
        img_dim=384, hidden=512, num_classes=args.num_classes,
        alpha=args.alpha, learnable_alpha=learnable_alpha, dropout=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_class_weight:
        cw = compute_class_weights_from_train(args.splits_csv, args.num_classes, device)
        ce = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    else:
        ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    ensure_dir(os.path.dirname(args.save_img_mlp))

    # 新增：记录训练历史的列表
    train_losses = []
    train_accs = []
    train_f1s = []
    val_accs = []
    val_f1s = []

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            total += loss.item()

        avg_train_loss = total / len(train_loader)
        train_losses.append(avg_train_loss)

        # 评估训练集和验证集
        train_acc, train_f1, _, _ = evaluate_and_record(model, train_loader, device, title=f"TRAIN epoch={epoch:02d}")
        val_acc, val_f1, _, _ = evaluate_and_record(model, val_loader, device, title=f"VAL epoch={epoch:02d}")

        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | "
              f"train_acc={train_acc:.4f} | train_f1={train_f1:.4f} | "
              f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            torch.save({
                "epoch": best_epoch,
                "best_val_macroF1": best_val_f1,
                "state_dict": best_state,
                "args": vars(args),
            }, args.save_img_mlp)

            print(f"[Saved best] {args.save_img_mlp} (epoch={best_epoch}, val_macroF1={best_val_f1:.4f})")

    # 绘制训练曲线
    if args.save_plots and len(train_losses) > 0:
        plot_training_curves(train_losses, train_accs, val_accs, train_f1s, val_f1s, args.plots_dir)

    # test best
    ckpt = torch.load(args.save_img_mlp, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    test_acc, test_f1, test_cm = eval_cls(model, test_loader, device, title="TEST (best checkpoint)")

    # 保存测试集混淆矩阵
    if args.save_plots:
        cm_path = os.path.join(args.plots_dir, "test_confusion_matrix.png")
        class_names = [f"Class {i}" for i in range(args.num_classes)]
        plot_confusion_matrix(test_cm,
                              title=f"Test Set Confusion Matrix (Best Epoch {best_epoch})",
                              save_path=cm_path,
                              class_names=class_names)
        print(f"[Visualization] Confusion matrix saved to: {cm_path}")

    print("\nDone. Best epoch:", ckpt["epoch"], "Best val macroF1:", ckpt["best_val_macroF1"])
    print("Saved:", args.save_img_mlp)


if __name__ == "__main__":
    main()