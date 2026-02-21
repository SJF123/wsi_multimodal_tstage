import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 你现在的 expanded + label 文件（按你前面合并方案生成的）
CLINICAL_CSV = "data/clinical_patient_expanded_labeled.csv"
SPLITS_LABELED = "data/splits_labeled.csv"

PATIENT_COL = "patient_id"
LABEL_COL = "label"   # 这里要求是 0..3
NUM_CLASSES = 4

# 你想“当作结构化特征”的列（其余列会自动识别）
# 如果某些列不存在，会自动跳过
NUMERIC_COLS_CAND = [
    "age_at_index", "age_at_diagnosis", "year_of_diagnosis"
]
CATEGORICAL_COLS_CAND = [
    "gender", "race", "ethnicity",
    "icd_10_code", "primary_diagnosis", "morphology",
    "tissue_or_organ_of_origin", "tumor_grade", "classification_of_tumor",
    "prior_malignancy", "prior_treatment", "synchronous_malignancy",
    "site_of_resection_or_biopsy"
]

def build_patient_split_map():
    df = pd.read_csv(SPLITS_LABELED)
    df[PATIENT_COL] = df[PATIENT_COL].astype(str).str.strip().str.upper()
    return df.groupby(PATIENT_COL)["split"].first().to_dict()

class TabularDataset(Dataset):
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
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
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
            pred = logits.argmax(dim=1).cpu().tolist()
            ys += list(y)
            ps += pred
    acc = accuracy_score(ys, ps)
    macro_f1 = f1_score(ys, ps, average="macro")
    return acc, macro_f1

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_csv(CLINICAL_CSV)
    df[PATIENT_COL] = df[PATIENT_COL].astype(str).str.strip().str.upper()

    # label 必须是 0..3
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df[df[LABEL_COL].isin([0,1,2,3])].copy()
    df["y"] = df[LABEL_COL].astype(int)

    # 贴 split
    p2s = build_patient_split_map()
    df["split"] = df[PATIENT_COL].map(p2s)
    df = df.dropna(subset=["split"]).copy()

    # 选取存在的列
    num_cols = [c for c in NUMERIC_COLS_CAND if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS_CAND if c in df.columns]

    print("Using numeric cols:", num_cols)
    print("Using categorical cols:", cat_cols)

    # numeric 清洗
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    # categorical 清洗 + one-hot
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["nan", "None", ""]), c] = "UNK"
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    # 标准化 numeric（只用 train）
    train_mask = df["split"] == "train"
    for c in num_cols:
        mu = df.loc[train_mask, c].mean()
        sd = df.loc[train_mask, c].std() + 1e-6
        df[c] = (df[c] - mu) / sd

    drop_cols = {PATIENT_COL, "split", LABEL_COL, "y"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    print("Patients rows Train/Val/Test:", len(train_df), len(val_df), len(test_df))
    print("Feature dim:", len(feature_cols))

    train_loader = DataLoader(TabularDataset(train_df, feature_cols), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TabularDataset(val_df, feature_cols), batch_size=256, shuffle=False)
    test_loader  = DataLoader(TabularDataset(test_df, feature_cols), batch_size=256, shuffle=False)

    model = MLP(in_dim=len(feature_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_state, best_val_f1 = None, -1

    for epoch in range(1, 200):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)  # 修复你之前的 warning
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
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, test_f1 = eval_model(model, test_loader, device)
    print("\n=== Best model on val (by macro-F1) ===")
    print(f"Test acc={test_acc:.4f} | Test macro-F1={test_f1:.4f}")

if __name__ == "__main__":
    main()
