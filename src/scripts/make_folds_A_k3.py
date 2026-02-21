# wsi_multimodal_tstage/src/scripts/make_folds_A_k3.py
import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/splits_labeled.csv", help="slide-level csv with columns incl. patient_id, split, label")
    ap.add_argument("--out_dir", default="data/folds_A_k3", help="output dir")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    assert "patient_id" in df.columns, "in_csv must contain patient_id"
    assert "split" in df.columns, "in_csv must contain split"
    assert "label" in df.columns, "in_csv must contain label"

    # 固定 test 不动：只在 train+val 上做 K 折
    test_df = df[df["split"] == "test"].copy()
    trainval_df = df[df["split"].isin(["train", "val"])].copy()

    # 以 patient 为单位做分层：先汇总每个 patient 的 label（mode）
    patient_tbl = (
        trainval_df.groupby("patient_id")["label"]
        .agg(lambda x: x.value_counts().index[0])  # 众数
        .reset_index()
        .rename(columns={"label": "patient_label"})
    )

    patients = patient_tbl["patient_id"].values
    y = patient_tbl["patient_label"].values

    # 分层 K 折（patient-level）
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    fold_maps = {}  # patient_id -> fold_id
    for fold_id, (_, val_idx) in enumerate(skf.split(patients, y)):
        for pid in patients[val_idx]:
            fold_maps[pid] = fold_id

    # 为 trainval_df 增加 fold_id
    trainval_df["fold_id"] = trainval_df["patient_id"].map(fold_maps)

    # 生成每个 fold 的 csv：val = fold_id==k, train = other folds；test 保持 test
    for fold in range(args.k):
        fold_train = trainval_df[trainval_df["fold_id"] != fold].copy()
        fold_val   = trainval_df[trainval_df["fold_id"] == fold].copy()

        fold_train["split"] = "train"
        fold_val["split"] = "val"

        out_df = pd.concat([fold_train, fold_val, test_df], ignore_index=True)

        out_path = os.path.join(args.out_dir, f"fold{fold}.csv")
        out_df.to_csv(out_path, index=False)

        # 打印一下每折分布，方便你检查类别是否极端
        print(f"\nSaved: {out_path}")
        print("Slide count by split:")
        print(out_df["split"].value_counts())

        print("Slide label dist (overall):")
        print(out_df["label"].value_counts().sort_index())

        print("Val label dist:")
        print(out_df[out_df["split"] == "val"]["label"].value_counts().sort_index())

    print("\nDone. (Scheme A: fixed test, CV on train+val)")

if __name__ == "__main__":
    main()
