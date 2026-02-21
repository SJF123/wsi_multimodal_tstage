from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset


class WSIPatchFeatureDataset(Dataset):
    """
    Image-only dataset:
    - reads data/splits_labeled.csv
    - loads features/vit_patch_feats/<slide_id>.pt  -> [200, 384]
    - returns: (x_img, y, slide_id, patient_id)
    """

    def __init__(
        self,
        splits_csv: str = "data/splits_labeled.csv",
        split: str = "train",
        feat_dir: str = "features/vit_patch_feats",
    ):
        self.splits_csv = splits_csv
        self.split = split
        self.feat_dir = Path(feat_dir)

        df = pd.read_csv(splits_csv)
        if "split" not in df.columns or "slide_id" not in df.columns or "patient_id" not in df.columns or "label" not in df.columns:
            raise ValueError("splits_labeled.csv must contain columns: split, slide_id, patient_id, label")

        df = df[df["split"] == split].copy()
        df["slide_id"] = df["slide_id"].astype(str)
        df["patient_id"] = df["patient_id"].astype(str)
        df["label"] = df["label"].astype(int)

        # 只保留特征文件真实存在的样本（稳妥：避免训练时找不到 .pt）
        feat_paths = []
        keep_rows = []
        for i, r in df.iterrows():
            p = self.feat_dir / f"{r['slide_id']}.pt"
            if p.exists():
                feat_paths.append(p)
                keep_rows.append(r)
        if len(keep_rows) == 0:
            raise RuntimeError(f"No feature files found for split={split}. Check feat_dir={feat_dir}")

        self.df = pd.DataFrame(keep_rows).reset_index(drop=True)
        self.feat_paths = feat_paths

        # 打印一个小摘要，方便你确认没问题
        print(f"[Dataset] split={split} samples={len(self.df)} slides={self.df['slide_id'].nunique()} patients={self.df['patient_id'].nunique()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x_img = torch.load(self.feat_paths[idx])  # [N, 384], N ~ 200
        y = int(row["label"])
        slide_id = row["slide_id"]
        patient_id = row["patient_id"]

        # 确保 dtype
        if not torch.is_tensor(x_img):
            x_img = torch.tensor(x_img)
        x_img = x_img.float()

        return x_img, y, slide_id, patient_id
