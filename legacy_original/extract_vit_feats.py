# 用于抽取patch+ViT特征，自动找出有组织的区域，随机采样 200 个 patch（224×224），
# 用 ViT 抽特征，保存成，features/vit_patch_feats/<slide_id>.pt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import os
from pathlib import Path
import random

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# =========================
# 基本参数（你现在阶段不用改）
# =========================
NUM_PATCHES = 200
PATCH_SIZE = 224
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# 路径设置
# =========================
SLIDES_CSV = "data/slides_400_matched.csv"
OUT_DIR = Path("features/vit_patch_feats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# ViT 模型（ImageNet 预训练）
# =========================
def load_vit():
    model = torch.hub.load(
        "facebookresearch/dino:main",
        "dino_vits16"
    )
    model.eval()
    model.to(DEVICE)
    return model

# =========================
# 图像预处理
# =========================
transform = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# =========================
# 简单组织区域判断（过滤背景）
# =========================
def is_tissue(patch, thresh=0.8):
    """
    patch: PIL Image
    """
    gray = np.array(patch.convert("L"))
    # 白色背景通常接近255
    return (gray < 240).mean() > (1 - thresh)

# =========================
# 从大图中随机采样 patch
# =========================
def sample_patches(img, num_patches):
    w, h = img.size
    patches = []
    tries = 0

    while len(patches) < num_patches and tries < num_patches * 20:
        x = random.randint(0, max(0, w - PATCH_SIZE))
        y = random.randint(0, max(0, h - PATCH_SIZE))
        patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
        if is_tissue(patch):
            patches.append(patch)
        tries += 1

    return patches

# =========================
# 主流程
# =========================
def main():
    slides = pd.read_csv(SLIDES_CSV)
    print("Total slides:", len(slides))

    model = load_vit()

    with torch.no_grad():
        for idx, row in slides.iterrows():
            slide_id = row["slide_id"]
            png_path = row["png_path"]

            out_path = OUT_DIR / f"{slide_id}.pt"
            if out_path.exists():
                print(f"[Skip] {slide_id}")
                continue

            print(f"[{idx+1}/{len(slides)}] Processing {slide_id}")

            try:
                img = Image.open(png_path).convert("RGB")
            except Exception as e:
                print("  Failed to open image:", e)
                continue

            patches = sample_patches(img, NUM_PATCHES)

            if len(patches) == 0:
                print("  No valid patches found, skip.")
                continue

            feats = []
            for p in patches:
                x = transform(p).unsqueeze(0).to(DEVICE)
                f = model(x)
                feats.append(f.squeeze(0).cpu())

            feats = torch.stack(feats)  # [N, 384]
            torch.save(feats, out_path)

    print("Feature extraction finished.")

if __name__ == "__main__":
    main()
