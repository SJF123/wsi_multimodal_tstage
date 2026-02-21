import pandas as pd
from pathlib import Path

EXPANDED = "data/clinical_patient_expanded_400_matched.csv"
SPLITS_LABELED = "data/splits_labeled.csv"
OUT = "data/clinical_patient_expanded_labeled.csv"

def main():
    cli = pd.read_csv(EXPANDED)
    splits = pd.read_csv(SPLITS_LABELED)

    # 统一 patient_id
    cli["patient_id"] = cli["patient_id"].astype(str).str.strip().str.upper()
    splits["patient_id"] = splits["patient_id"].astype(str).str.strip().str.upper()

    # 从 splits 得到 patient-level label（0..3）
    p_label = splits.groupby("patient_id")["label"].first().reset_index()

    # 合并
    merged = cli.merge(p_label, on="patient_id", how="left")

    # 检查有没有没合到 label 的
    miss = merged["label"].isna().sum()
    if miss > 0:
        ex = merged.loc[merged["label"].isna(), "patient_id"].unique().tolist()[:10]
        print(f"WARNING: {miss} patients have no label match. Examples:", ex)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("Saved:", OUT)
    print("Rows:", len(merged))
    print("Label distribution:")
    print(merged["label"].value_counts(dropna=False).sort_index())

if __name__ == "__main__":
    main()
