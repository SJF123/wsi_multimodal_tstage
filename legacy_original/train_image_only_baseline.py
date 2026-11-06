'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
'''

# image-only 做的仅仅是图像的基线实验
# python -m src.scripts.train_image_only_baseline


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datasets.feature_dataset import WSIPatchFeatureDataset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from collections import Counter


class MeanPoolClassifier(nn.Module):
    def __init__(self, in_dim=384, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):  # x: [B, N, D]
        x = x.mean(dim=1)   # [B, D]
        return self.fc(x)   # [B, C]


def eval_one_epoch(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, *_ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            ys.extend(y.cpu().tolist())
            ps.extend(pred.cpu().tolist())
    acc = accuracy_score(ys, ps)
    macro_f1 = f1_score(ys, ps, average="macro")
    return acc, macro_f1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = WSIPatchFeatureDataset(split="train")
    val_ds = WSIPatchFeatureDataset(split="val")
    test_ds = WSIPatchFeatureDataset(split="test")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = MeanPoolClassifier(in_dim=384, num_classes=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val_f1 = -1
    best_state = None

    for epoch in range(1, 200):
        model.train()
        total_loss = 0.0

        for x, y, *_ in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        val_acc, val_f1 = eval_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={total_loss/len(train_loader):.4f} | val_acc={val_acc:.4f} | val_macroF1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # 测试
    model.load_state_dict(best_state)
    test_acc, test_f1 = eval_one_epoch(model, test_loader, device)
    print("\n=== Best model on val (by macro-F1) ===")
    print(f"Test acc={test_acc:.4f} | Test macro-F1={test_f1:.4f}")


if __name__ == "__main__":
    main()
