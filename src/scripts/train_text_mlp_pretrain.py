# train_text_mlp_pretrain.py
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 兼容 “python src/scripts/xxx.py” 直接运行
# ---------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
    if path and len(path) > 0:
        os.makedirs(path, exist_ok=True)


# -----------------------------
# Dataset
# -----------------------------
class PatientTextEmbDataset(Dataset):
    def __init__(self, splits_csv, text_emb_pt, split,
                 patient_col="patient_id", label_col="label_img"):
        super().__init__()

        df = pd.read_csv(splits_csv)
        df = df[df["split"] == split].copy()

        emb_dict = torch.load(text_emb_pt, map_location="cpu")

        rows = []
        for _, r in df.iterrows():
            pid = str(r[patient_col])
            if pid in emb_dict:
                rows.append((emb_dict[pid].float(), int(r[label_col])))

        self.rows = rows
        print(f"[Dataset] split={split} samples={len(self.rows)}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        x, y = self.rows[idx]
        return x, torch.tensor(y, dtype=torch.long)


# -----------------------------
# Model
# -----------------------------
class TextMLPClassifier(nn.Module):
    def __init__(self, cli_dim=768, hidden=1024, num_classes=4):
        super().__init__()
        self.cli_ft = ResidualMLP(cli_dim, hidden=hidden)
        self.head = nn.Linear(cli_dim, num_classes)

    def forward(self, x):
        f = self.cli_ft(x)
        logits = self.head(f)
        return logits


@torch.no_grad()
def eval_cls(model, loader, device):
    model.eval()
    ys, ps = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)

        ys.extend(y.cpu().tolist())
        ps.extend(pred.cpu().tolist())

    acc = accuracy_score(ys, ps)
    macro_f1 = f1_score(ys, ps, average="macro")

    return acc, macro_f1


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--splits_csv", required=True)
    ap.add_argument("--text_emb_pt", required=True)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--plots_dir", default="plots/text_pretrain")
    ap.add_argument("--metrics_xlsx", default="")

    args = ap.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ensure_dir(args.plots_dir)

    # Dataset
    train_ds = PatientTextEmbDataset(args.splits_csv, args.text_emb_pt, "train")
    val_ds   = PatientTextEmbDataset(args.splits_csv, args.text_emb_pt, "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)

    # Model
    model = TextMLPClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    # History
    train_losses = []
    train_accs = []
    train_f1s = []
    val_accs = []
    val_f1s = []

    # Training
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)

        train_acc, train_f1 = eval_cls(model, train_loader, device)
        val_acc, val_f1     = eval_cls(model, val_loader, device)

        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch:04d} | loss={avg_loss:.4f} | "
              f"train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

    # -----------------------------
    # Save metrics to Excel
    # -----------------------------
    out_xlsx = args.metrics_xlsx.strip()
    if out_xlsx == "":
        out_xlsx = os.path.join(args.plots_dir, "metrics.xlsx")

    df_metrics = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "loss": train_losses,
        "train_acc": train_accs,
        "train_macroF1": train_f1s,
        "val_acc": val_accs,
        "val_macroF1": val_f1s,
    })

    df_metrics.to_excel(out_xlsx, index=False)
    print(f"\n[Saved] metrics excel -> {out_xlsx}")


if __name__ == "__main__":
    main()
# python src/scripts/train_text_mlp_pretrain.py --splits_csv data/splits_with_clinical.csv --text_emb_pt features/clinical_feats/clinical_text_emb_distilbert_patient.pt --epochs 1000 --batch_size 32 --metrics_xlsx plots/text_pretrain/run2_metrics.xlsx