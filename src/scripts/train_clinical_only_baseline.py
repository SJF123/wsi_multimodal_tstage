import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CLINICAL_PATIENT = "data/clinical_patient_expanded_400_matched.csv"
SPLITS_LABELED = "data/splits_labeled.csv"
NUM_CLASSES = 4  # T1~T4 -> 0~3

# 你的真实列名（已对齐）
PATIENT_COL = "patient_id"
LABEL_COL = "label"          # 1..4
AGE_COL = "age_at_index"
SEX_COL = "gender"
RACE_COL = "race"

LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}


def build_patient_split_map():
    df = pd.read_csv(SPLITS_LABELED)
    # patient_id -> split（病人级）
    return df.groupby(PATIENT_COL)["split"].first().to_dict()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[PATIENT_COL] = df[PATIENT_COL].astype(str).str.strip()

    # label: 1..4 -> 0..3
    df["y"] = df[LABEL_COL].astype(int).map(LABEL_MAP)
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    # age 数值化 + 缺失填充（你这里缺失率0，但保留稳妥）
    df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
    df[AGE_COL] = df[AGE_COL].fillna(df[AGE_COL].median())

    # 类别特征缺失处理（你这里缺失率0，但保留稳妥）
    for c in [SEX_COL, RACE_COL]:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["nan", "None", ""]), c] = "UNK"

    return df


class ClinicalDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.loc[idx, self.feature_cols].values.astype(np.float32)
        y = int(self.df.loc[idx, "y"])
        return torch.tensor(x), y


class MLP(nn.Module):
    def __init__(self, in_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def eval_model(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            ys += list(y)
            ps += pred
    acc = accuracy_score(ys, ps)
    macro_f1 = f1_score(ys, ps, average="macro")
    return acc, macro_f1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_csv(CLINICAL_PATIENT)
    df = preprocess(df)

    p2s = build_patient_split_map()
    df["split"] = df[PATIENT_COL].map(p2s)
    df = df.dropna(subset=["split"]).copy()

    # one-hot 类别特征
    df = pd.get_dummies(df, columns=[SEX_COL, RACE_COL], dummy_na=False)

    # 标准化 age（只用 train 的均值方差）
    train_mask = df["split"] == "train"
    mu = df.loc[train_mask, AGE_COL].mean()
    sd = df.loc[train_mask, AGE_COL].std() + 1e-6
    df[AGE_COL] = (df[AGE_COL] - mu) / sd

    drop_cols = {PATIENT_COL, "split", LABEL_COL, "y"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("Patients:", len(df), "Train/Val/Test:", len(train_df), len(val_df), len(test_df))
    print("Feature dim:", len(feature_cols))
    print("Feature cols:", feature_cols)

    train_loader = DataLoader(ClinicalDataset(train_df, feature_cols), batch_size=32, shuffle=True)
    val_loader = DataLoader(ClinicalDataset(val_df, feature_cols), batch_size=64, shuffle=False)
    test_loader = DataLoader(ClinicalDataset(test_df, feature_cols), batch_size=64, shuffle=False)

    model = MLP(in_dim=len(feature_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_state, best_val_f1 = None, -1

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            #y = torch.tensor(y).to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        val_acc, val_f1 = eval_model(model, val_loader, device)
        print(f"Epoch {epoch:02d} | loss={total_loss/len(train_loader):.4f} | val_acc={val_acc:.4f} | val_macroF1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, test_f1 = eval_model(model, test_loader, device)
    print("\n=== Best model on val (by macro-F1) ===")
    print(f"Test acc={test_acc:.4f} | Test macro-F1={test_f1:.4f}")


if __name__ == "__main__":
    main()
