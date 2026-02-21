# 核心功能是将病理图像特征与临床文本特征对齐并打包，以供多模态模型训练
import os
import torch
import pandas as pd

def norm_pid(x: str) -> str:
    return str(x).strip().upper()

class CLIPImgTextDataset(torch.utils.data.Dataset):
    """
    Returns:
      patch_feats: [N, D] float32  (e.g., [200,384])
      text_emb:    [768] float32
      y:           int (0..3)
    """
    def __init__(
        self,
        splits_csv="data/splits_labeled.csv",
        text_emb_pt="data/clinical_text_emb_distilbert.pt",
        feat_dir="features/vit_patch_feats",
        split="train",
        patient_col="patient_id",
        slide_col="slide_id",
        label_col="label",
        max_patches=200,
    ):
        self.df = pd.read_csv(splits_csv)
        self.df[patient_col] = self.df[patient_col].apply(norm_pid)
        self.df = self.df[self.df["split"] == split].copy()

        # slide_id 列名兼容
        if slide_col not in self.df.columns:
            for alt in ["slide", "slide_name", "filename", "png_name"]:
                if alt in self.df.columns:
                    self.df = self.df.rename(columns={alt: slide_col})
                    break
        if slide_col not in self.df.columns:
            raise ValueError("splits_labeled.csv must contain a slide_id (or similar) column.")

        self.text_emb = torch.load(text_emb_pt, map_location="cpu")  # dict pid -> [768]
        self.feat_dir = feat_dir
        self.patient_col = patient_col
        self.slide_col = slide_col
        self.label_col = label_col
        self.max_patches = max_patches

        self.samples = []
        for _, r in self.df.iterrows():
            pid = r[patient_col]
            slide = str(r[slide_col])
            y = int(r[label_col])

            if pid not in self.text_emb:
                continue
            feat_path = os.path.join(feat_dir, f"{slide}.pt")
            if not os.path.exists(feat_path):
                continue
            self.samples.append((feat_path, pid, y))

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check slide_id naming vs .pt filenames, and text_emb patient ids.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, pid, y = self.samples[idx]
        feats = torch.load(feat_path, map_location="cpu")
        if isinstance(feats, dict) and "feats" in feats:
            feats = feats["feats"]
        feats = feats.float()  # [N,D]

        if feats.size(0) > self.max_patches:
            feats = feats[: self.max_patches]

        t = self.text_emb[pid].float()  # [768]
        return feats, t, int(y)
