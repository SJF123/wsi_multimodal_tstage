# 用于标签映射
# T1→0
# T2→1
# T3→2
# T4→3
import pandas as pd
from pathlib import Path

SPLITS_CSV = "data/splits.csv"
CLINICAL_PATIENT_CSV = "data/clinical_patient_400_matched.csv"
OUT_CSV = "data/splits_labeled.csv"

# 4类：T1~T4 映射到 0~3
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}

def main():
    print("Reading:", SPLITS_CSV)
    splits = pd.read_csv(SPLITS_CSV)

    print("Reading:", CLINICAL_PATIENT_CSV)
    clinical = pd.read_csv(CLINICAL_PATIENT_CSV)

    # 必要列检查
    for c in ["patient_id"]:
        if c not in splits.columns:
            raise ValueError(f"splits.csv missing column: {c}")
    for c in ["patient_id", "label"]:
        if c not in clinical.columns:
            raise ValueError(f"clinical_patient_400_matched.csv missing column: {c}")

    # patient_id 对齐（防止空格大小写问题）
    splits["patient_id"] = splits["patient_id"].astype(str).str.strip()
    clinical["patient_id"] = clinical["patient_id"].astype(str).str.strip()

    # patient-level label 映射到 splits
    p2label = clinical.set_index("patient_id")["label"].to_dict()
    splits["label_raw"] = splits["patient_id"].map(p2label)

    # 检查未匹配
    n_missing = splits["label_raw"].isna().sum()
    if n_missing > 0:
        missing_p = splits.loc[splits["label_raw"].isna(), "patient_id"].unique().tolist()
        print(f"WARNING: {n_missing} rows have no label match. Example patients:", missing_p[:10])

    # 映射到 0~3
    splits["label"] = splits["label_raw"].map(LABEL_MAP)

    # 检查非法标签
    bad = splits[splits["label"].isna()]
    if len(bad) > 0:
        print("WARNING: Some labels are not in LABEL_MAP (expected raw in {1,2,3,4}).")
        print("Bad raw labels example:", bad["label_raw"].dropna().unique()[:10])
        # 可以选择直接删除这些行
        splits = splits.dropna(subset=["label"]).copy()

    splits["label"] = splits["label"].astype(int)

    # 保存
    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_path, index=False)

    print("Saved:", OUT_CSV)
    print("Label distribution (slide-level):")
    print(splits["label"].value_counts().sort_index())
    print("\nLabel distribution by split:")
    print(splits.groupby("split")["label"].value_counts().sort_index())

if __name__ == "__main__":
    main()
