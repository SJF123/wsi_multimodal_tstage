import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image, ImageFile

# =========================
# PIL / 安全设置（关键）
# =========================
Image.MAX_IMAGE_PIXELS = None                 # 允许超大图
ImageFile.LOAD_TRUNCATED_IMAGES = True        # 容忍部分截断图片
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# =========================
# 可调参数（新手建议默认不改）
# =========================
SEED = 42
NUM_PATCHES = 200
PATCH_SIZE = 224
BATCH_SIZE = 32          # 显存够可改 64 更快
MAX_SIDE = 8000          # 超大图先缩到最长边 <= 8000；内存小可改 6000
MIN_NONWHITE = 0.15      # 组织区域判定：灰度<阈值的比例至少多少
WHITE_THR = 240          # 背景判定阈值（越大越严格）

# =========================
# 路径
# =========================
SLIDES_CSV = "data/slides_400_matched.csv"
MISSING_CSV = "logs/missing_slides.csv"
OUT_DIR = Path("features/vit_patch_feats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAIL_LOG = Path("logs/failed_images.txt")
FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# ViT (DINO) 加载
# =========================
def load_vit():
    # 你之前已经跑通过这个缓存，这里继续复用
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
    model.eval().to(DEVICE)
    return model

# =========================
# patch 预处理
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
# 组织区域判定（过滤白背景）
# =========================
def is_tissue(patch: Image.Image,
              white_thr: int = WHITE_THR,
              min_nonwhite: float = MIN_NONWHITE) -> bool:
    gray = np.array(patch.convert("L"))
    nonwhite_ratio = (gray < white_thr).mean()
    return nonwhite_ratio >= min_nonwhite

# =========================
# 安全打开 + 缩小（关键：先缩后 convert）
# =========================
def safe_open_and_downscale(png_path: str, max_side: int = MAX_SIDE) -> Image.Image:
    """
    关键原则：不要对原始超大图做 load/convert/copy，这会非常慢甚至卡死。
    先用 thumbnail 缩小到可处理尺寸，再 convert。
    """
    img = Image.open(png_path)

    # 先缩小，避免后续 convert 触发超大 copy
    img.thumbnail((max_side, max_side), resample=Image.BILINEAR)

    # 缩小后再转 RGB 更稳定
    img = img.convert("RGB")
    return img

# =========================
# 随机采样 patch
# =========================
def sample_patches(img: Image.Image, num_patches: int) -> list:
    w, h = img.size
    patches = []
    tries = 0
    max_tries = num_patches * 30

    if w < PATCH_SIZE or h < PATCH_SIZE:
        return patches

    while len(patches) < num_patches and tries < max_tries:
        x = random.randint(0, w - PATCH_SIZE)
        y = random.randint(0, h - PATCH_SIZE)
        patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
        if is_tissue(patch):
            patches.append(patch)
        tries += 1

    return patches

# =========================
# 批量提特征（加速）
# =========================
def vit_encode_patches(model, patches: list) -> torch.Tensor:
    xs = torch.stack([transform(p) for p in patches], dim=0)  # [N,3,224,224]
    feats_list = []

    with torch.no_grad():
        for i in range(0, xs.size(0), BATCH_SIZE):
            xb = xs[i:i + BATCH_SIZE].to(DEVICE, non_blocking=True)
            fb = model(xb).cpu()  # [B,384]
            feats_list.append(fb)

    feats = torch.cat(feats_list, dim=0)  # [N,384]
    return feats

# =========================
# 主流程
# =========================
def main():
    print("Device:", DEVICE)
    print("Reading:", SLIDES_CSV)
    slides = pd.read_csv(SLIDES_CSV)

    print("Reading:", MISSING_CSV)
    miss = pd.read_csv(MISSING_CSV)
    miss_ids = set(miss["slide_id"].astype(str))

    todo = slides[slides["slide_id"].astype(str).isin(miss_ids)].copy()
    print("Missing slides to process:", len(todo))

    model = load_vit()

    # 逐个 slide 处理
    for idx, row in enumerate(todo.itertuples(index=False), start=1):
        slide_id = str(row.slide_id)
        png_path = str(row.png_path)

        out_path = OUT_DIR / f"{slide_id}.pt"
        if out_path.exists():
            print(f"[Skip] {slide_id}")
            continue

        print(f"[{idx}/{len(todo)}] Processing {slide_id}")

        # 1) 打开并缩小图片
        try:
            img = safe_open_and_downscale(png_path, max_side=MAX_SIDE)
        except Exception as e:
            with open(FAIL_LOG, "a", encoding="utf-8") as f:
                f.write(f"{slide_id}\t{png_path}\tOPEN_FAIL\t{repr(e)}\n")
            print("  Failed to open/resize/convert:", repr(e))
            continue

        # 2) 采样 patch
        patches = sample_patches(img, NUM_PATCHES)
        if len(patches) == 0:
            with open(FAIL_LOG, "a", encoding="utf-8") as f:
                f.write(f"{slide_id}\t{png_path}\tNO_VALID_PATCHES\n")
            print("  No valid patches found, skip.")
            continue

        # 3) ViT 提特征
        try:
            feats = vit_encode_patches(model, patches)  # [N,384]
        except Exception as e:
            with open(FAIL_LOG, "a", encoding="utf-8") as f:
                f.write(f"{slide_id}\t{png_path}\tVIT_FAIL\t{repr(e)}\n")
            print("  ViT encoding failed:", repr(e))
            continue

        # 4) 保存
        try:
            torch.save(feats, out_path)
        except Exception as e:
            with open(FAIL_LOG, "a", encoding="utf-8") as f:
                f.write(f"{slide_id}\t{png_path}\tSAVE_FAIL\t{repr(e)}\n")
            print("  Save failed:", repr(e))
            continue

    print("\nDone. Now re-run:")
    print("  python src\\scripts\\summarize_feat_status.py")
    print("If needed, see fail log:")
    print(f"  {FAIL_LOG}")

if __name__ == "__main__":
    main()
