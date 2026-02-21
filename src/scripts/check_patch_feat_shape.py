import os
import torch
import random

# ========= 修改这里 =========
FEAT_DIR = "features/vit_patch_feats"
# ===========================

def main():
    files = [f for f in os.listdir(FEAT_DIR) if f.endswith(".pt")]
    print(f"Found {len(files)} .pt files in {FEAT_DIR}")

    if len(files) == 0:
        print("❌ No .pt files found!")
        return

    # 随机选一个文件
    fname = random.choice(files)
    path = os.path.join(FEAT_DIR, fname)

    print("\n[Inspect file]")
    print("File name:", fname)

    data = torch.load(path, map_location="cpu")

    print("\n[Loaded object type]")
    print(type(data))

    # 情况 1：直接是 Tensor
    if isinstance(data, torch.Tensor):
        print("\n[Case A] Tensor directly saved")
        print("Shape:", data.shape)
        print("Dtype:", data.dtype)

    # 情况 2：是 dict
    elif isinstance(data, dict):
        print("\n[Case B] Dict saved, keys:")
        for k in data.keys():
            print(f"  - key: {k}, type: {type(data[k])}")
            if isinstance(data[k], torch.Tensor):
                print(f"    shape: {data[k].shape}, dtype: {data[k].dtype}")

    # 其他情况
    else:
        print("\n[Case C] Other type")
        print(data)

if __name__ == "__main__":
    main()
