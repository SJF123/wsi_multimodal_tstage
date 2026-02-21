import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt


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
    if path and len(path) > 0:
        os.makedirs(path, exist_ok=True)


def compute_class_weights_from_train(splits_csv: str, num_classes: int, device: str):
    df = pd.read_csv(splits_csv)
    train_df = df[df["split"] == "train"]
    counts = train_df["label"].value_counts().sort_index()

    full_counts = np.zeros(num_classes, dtype=np.int64)
    for k, v in counts.items():
        k = int(k)
        if 0 <= k < num_classes:
            full_counts[k] = int(v)

    print("[Train label counts]", {i: int(full_counts[i]) for i in range(num_classes)})

    w = 1.0 / (full_counts.astype(np.float32) + 1e-6)
    w = w / w.mean()
    class_weights = torch.tensor(w, dtype=torch.float32).to(device)
    print("[Class weights]", class_weights.detach().cpu().numpy())
    return class_weights


# -----------------------------
# SCI-style plotting helpers
# -----------------------------
def sci_style():
    """
    Lightweight SCI-ish style:
    - clean spines
    - subtle grid
    - consistent dpi
    """
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
    })


def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


# Your requested contrast colors
COLOR_A = rgb_to_hex(179, 212, 155)  # greenish
COLOR_B = rgb_to_hex(205, 163, 203)  # purplish
COLOR_NEUTRAL = "#4d4d4d"


def plot_curve(
    x, y_list, labels, colors, title, xlabel, ylabel, save_path,
    marker=True
):
    sci_style()
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = plt.gca()

    for y, lab, col in zip(y_list, labels, colors):
        if marker:
            ax.plot(x, y, label=lab, linewidth=2.2, color=col, marker="o", markersize=4)
        else:
            ax.plot(x, y, label=lab, linewidth=2.2, color=col)

    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.02)

    # subtle axis lines
    ax.spines["left"].set_alpha(0.8)
    ax.spines["bottom"].set_alpha(0.8)

    ensure_dir(os.path.dirname(save_path))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, title, save_path, cmap="Purples"):
    sci_style()
    fig = plt.figure(figsize=(6.0, 5.2))
    ax = plt.gca()

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title, pad=10)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    # annotate cells
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.clip(cm_sum, 1, None)

    thresh = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val}\n({cm_norm[i,j]*100:.1f}%)"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=9,
                color="white" if val > thresh else "black"
            )

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ensure_dir(os.path.dirname(save_path))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, verbose=False, title=""):
    model.eval()
    ys, ps = [], []
    for patch, y in loader:
        patch = patch.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(patch)
        pred = logits.argmax(dim=1)
        ys.extend(y.cpu().tolist())
        ps.extend(pred.cpu().tolist())

    acc = accuracy_score(ys, ps)
    macro_f1 = f1_score(ys, ps, average="macro")

    if verbose:
        if title:
            print(f"\n===== {title} =====")
        print("[Confusion Matrix]")
        print(confusion_matrix(ys, ps))
        print("\n[Classification Report]")
        print(classification_report(ys, ps, digits=4))

    return acc, macro_f1


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    ys, ps = [], []
    for patch, y in loader:
        patch = patch.to(device, non_blocking=True)
        logits = model(patch)
        pred = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        ys.extend(y.numpy().tolist())
        ps.extend(pred)
    return np.array(ys), np.array(ps)


# -----------------------------
# Dataset (Image-only)
# -----------------------------
class ImageOnlyPatchDataset(Dataset):
    def __init__(self, splits_csv, feat_dir, split="train", max_patches=200):
        self.df = pd.read_csv(splits_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.feat_dir = feat_dir
        self.max_patches = max_patches

        cand_cols = ["slide_id", "slide", "wsi_id", "filename"]
        self.id_col = None
        for c in cand_cols:
            if c in self.df.columns:
                self.id_col = c
                break
        if self.id_col is None:
            raise ValueError(
                f"splits_csv must contain one of {cand_cols}, but got columns: {list(self.df.columns)}"
            )

    def __len__(self):
        return len(self.df)

    def _feat_path(self, row):
        sid = str(row[self.id_col])
        sid = os.path.splitext(sid)[0]
        return os.path.join(self.feat_dir, f"{sid}.pt")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = int(row["label"])
        fp = self._feat_path(row)

        if not os.path.exists(fp):
            raise FileNotFoundError(f"Feature file not found: {fp}")

        feat = torch.load(fp, map_location="cpu")
        if isinstance(feat, dict) and "feat" in feat:
            feat = feat["feat"]

        if feat.dim() == 1:
            feat = feat.unsqueeze(0)

        # enforce max_patches
        if feat.size(0) > self.max_patches:
            feat = feat[: self.max_patches, :]
        elif feat.size(0) < self.max_patches:
            pad = torch.zeros(self.max_patches - feat.size(0), feat.size(1), dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=0)

        return feat.float(), torch.tensor(y, dtype=torch.long)


# -----------------------------
# Model: Attention Pooling + Residual MLP
# -----------------------------
class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden=512, dropout=0.1, alpha=0.2, learnable_alpha=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x):
        return x + self.alpha * self.mlp(x)


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        w = self.attn(x)            # [B, N, 1]
        w = torch.softmax(w, dim=1) # [B, N, 1]
        return (w * x).sum(dim=1)   # [B, D]


class ImagePretrainModel(nn.Module):
    def __init__(self, img_dim=384, num_classes=4, dropout=0.1, alpha_img=0.2, learnable_alpha=False):
        super().__init__()
        self.pool = AttentionPooling(img_dim)
        self.img_ft = ResidualMLP(img_dim, hidden=512, dropout=dropout, alpha=alpha_img, learnable_alpha=learnable_alpha)
        self.classifier = nn.Linear(img_dim, num_classes)

    def forward(self, patch_feats):
        x = self.pool(patch_feats)
        x = self.img_ft(x)
        logits = self.classifier(x)
        return logits


# -----------------------------
# DataLoader builder
# -----------------------------
def build_loader(dataset, batch_size, shuffle, args, drop_last=False):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=drop_last,
    )
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", default="data/splits_labeled.csv")
    ap.add_argument("--feat_dir", default="features/vit_patch_feats")
    ap.add_argument("--max_patches", type=int, default=200)
    ap.add_argument("--num_classes", type=int, default=4)

    ap.add_argument("--alpha_img", type=float, default=0.2)
    ap.add_argument("--alpha_mode", choices=["fixed", "learnable"], default="fixed")
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_path", default="checkpoints/image_mlp_pretrain.pt")
    ap.add_argument("--fig_dir", default="plots/image_pre_train")  # <-- plots go here
    ap.add_argument("--verbose_eval", action="store_true")
    ap.add_argument("--debug", action="store_true")

    # speed knobs
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")          # use if GPU
    ap.add_argument("--persistent_workers", action="store_true")  # windows卡住就别开
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--val_bs", type=int, default=256)
    ap.add_argument("--test_bs", type=int, default=256)

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("CUDA name:", torch.cuda.get_device_name(0))

    ensure_dir(os.path.dirname(args.save_path))
    ensure_dir(args.fig_dir)

    # Datasets
    train_ds = ImageOnlyPatchDataset(args.splits_csv, args.feat_dir, split="train", max_patches=args.max_patches)
    val_ds   = ImageOnlyPatchDataset(args.splits_csv, args.feat_dir, split="val",   max_patches=args.max_patches)
    test_ds  = ImageOnlyPatchDataset(args.splits_csv, args.feat_dir, split="test",  max_patches=args.max_patches)
    print("Samples Train/Val/Test:", len(train_ds), len(val_ds), len(test_ds))

    # Loaders
    train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True,  args=args, drop_last=True)
    # 为了计算“训练集acc/f1”，我们用一个不shuffle的大batch评估loader（更快、更稳定）
    train_eval_loader = build_loader(train_ds, batch_size=args.val_bs, shuffle=False, args=args, drop_last=False)

    val_loader   = build_loader(val_ds,   batch_size=args.val_bs,  shuffle=False, args=args, drop_last=False)
    test_loader  = build_loader(test_ds,  batch_size=args.test_bs, shuffle=False, args=args, drop_last=False)

    # Model
    learnable_alpha = (args.alpha_mode == "learnable")
    model = ImagePretrainModel(
        img_dim=384,
        num_classes=args.num_classes,
        dropout=args.dropout,
        alpha_img=args.alpha_img,
        learnable_alpha=learnable_alpha
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    class_weights = compute_class_weights_from_train(args.splits_csv, args.num_classes, device)
    ce = nn.CrossEntropyLoss(weight=class_weights)

    # History for visualization
    epochs = []
    train_losses = []
    train_accs, val_accs = [], []
    train_f1s,  val_f1s  = [], []

    best_val_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (patch, y) in enumerate(train_loader):
            patch = patch.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if args.debug and epoch == 1 and batch_idx == 0:
                print("[DEBUG] patch:", patch.shape, "y:", y.shape)

            logits = model(patch)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # Evaluate train/val each epoch (for curves)
        tr_acc, tr_f1 = evaluate(model, train_eval_loader, device, verbose=False)
        va_acc, va_f1 = evaluate(model, val_loader, device, verbose=args.verbose_eval, title=f"VAL epoch={epoch:02d}")

        epochs.append(epoch)
        train_losses.append(avg_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        train_f1s.append(tr_f1)
        val_f1s.append(va_f1)

        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | train_acc={tr_acc:.4f} | train_macroF1={tr_f1:.4f} | val_acc={va_acc:.4f} | val_macroF1={va_f1:.4f}")

        # Save best by val macro-F1
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            torch.save({
                "epoch": best_epoch,
                "best_val_macroF1": best_val_f1,
                "state_dict": best_state,
                "args": vars(args),
            }, args.save_path)

    # ---- Plot curves (SCI-ish) ----
    plot_curve(
        x=epochs,
        y_list=[train_losses],
        labels=["Train loss"],
        colors=[COLOR_NEUTRAL],
        title="Training Loss",
        xlabel="Epoch",
        ylabel="Cross-Entropy Loss",
        save_path=os.path.join(args.fig_dir, "train_loss.png"),
        marker=True
    )

    plot_curve(
        x=epochs,
        y_list=[train_accs, val_accs],
        labels=["Train Acc", "Val Acc"],
        colors=[COLOR_A, COLOR_B],
        title="Accuracy Curve",
        xlabel="Epoch",
        ylabel="Accuracy",
        save_path=os.path.join(args.fig_dir, "acc_train_val.png"),
        marker=True
    )

    plot_curve(
        x=epochs,
        y_list=[train_f1s, val_f1s],
        labels=["Train Macro-F1", "Val Macro-F1"],
        colors=[COLOR_A, COLOR_B],
        title="Macro-F1 Curve",
        xlabel="Epoch",
        ylabel="Macro-F1",
        save_path=os.path.join(args.fig_dir, "f1_train_val.png"),
        marker=True
    )

    # ---- Test with best checkpoint ----
    ckpt = torch.load(args.save_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # verbose print
    test_acc, test_f1 = evaluate(model, test_loader, device, verbose=True, title="TEST (best image-pretrain)")

    # confusion matrix plot
    y_true, y_pred = predict_all(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred)

    class_names = [str(i) for i in range(args.num_classes)]
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        title="Test Confusion Matrix",
        save_path=os.path.join(args.fig_dir, "test_confusion_matrix.png"),
        cmap="Purples"
    )

    print("\n=== Image-Only Pretrain (best on val macro-F1) ===")
    print(f"Best val macro-F1={ckpt['best_val_macroF1']:.4f} (epoch={ckpt['epoch']})")
    print(f"Test acc={test_acc:.4f} | Test macro-F1={test_f1:.4f}")
    print(f"Saved ckpt: {args.save_path}")
    print(f"Saved figs: {args.fig_dir}")


if __name__ == "__main__":
    main()
