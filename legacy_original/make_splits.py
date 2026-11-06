# 将数据按照患者id进行划分，再将划分结果映射到图片，这样能够保证数据不泄露
# 同一个患者的数据不会既出现在训练集又出现在训练集
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# =========================
# 配置
# =========================
SLIDES_CSV = "data/slides_400_matched.csv"
OUT_CSV = "data/splits.csv"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

# =========================
# 主流程
# =========================
def main():
    print("Reading:", SLIDES_CSV)
    slides = pd.read_csv(SLIDES_CSV)

    # 必须包含这些列
    required_cols = ["slide_id", "patient_id"]
    for c in required_cols:
        if c not in slides.columns:
            raise ValueError(f"Missing required column: {c}")

    # 1) 病人级去重
    patients = slides[["patient_id"]].drop_duplicates().reset_index(drop=True)
    print("Total patients:", len(patients))

    # 2) 先划分 train / temp
    train_p, temp_p = train_test_split(
        patients["patient_id"],
        test_size=(1 - TRAIN_RATIO),
        random_state=SEED,
        shuffle=True,
    )

    # 3) 再把 temp 划分成 val / test
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_p, test_p = train_test_split(
        temp_p,
        test_size=(1 - val_size),
        random_state=SEED,
        shuffle=True,
    )

    train_p = set(train_p)
    val_p = set(val_p)
    test_p = set(test_p)

    print(f"Train patients: {len(train_p)}")
    print(f"Val patients:   {len(val_p)}")
    print(f"Test patients:  {len(test_p)}")

    # 4) 映射回 slide
    def assign_split(pid):
        if pid in train_p:
            return "train"
        if pid in val_p:
            return "val"
        if pid in test_p:
            return "test"
        return "unknown"

    slides["split"] = slides["patient_id"].apply(assign_split)

    # 安全检查
    assert (slides["split"] != "unknown").all()

    # 5) 保存
    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slides.to_csv(out_path, index=False)

    print("Saved splits to:", out_path)
    print("\nSlide count by split:")
    print(slides["split"].value_counts())

    print("\nPatient count by split:")
    print(
        slides.groupby("split")["patient_id"]
        .nunique()
        .rename("num_patients")
    )

if __name__ == "__main__":
    main()
