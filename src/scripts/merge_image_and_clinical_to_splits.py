import os
import argparse
import pandas as pd


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_csv", default="data/图像数据标签.csv",
                    help="图像划分主表（包含 patient_id/slide_id/split/label 等）")
    ap.add_argument("--cli_csv", default="data/体征数据.csv",
                    help="体征数据表（包含 patient_id 和体征字段）")
    ap.add_argument("--out_csv", default="data/splits_with_clinical.csv",
                    help="输出合并后的CSV（保持图像划分不变）")

    ap.add_argument("--img_id_col", default="patient_id", help="图像表的ID列名")
    ap.add_argument("--cli_id_col", default="patient_id", help="体征表的ID列名")

    # 体征表里也可能有 label，是否保留（默认保留，改名为 clinical_label）
    ap.add_argument("--keep_clinical_label", action="store_true", default=True)

    args = ap.parse_args()

    # 读取
    img_df = pd.read_csv(args.img_csv)
    cli_df = pd.read_csv(args.cli_csv)

    # 检查ID列
    assert args.img_id_col in img_df.columns, f"[ERROR] img_csv 缺少列: {args.img_id_col}"
    assert args.cli_id_col in cli_df.columns, f"[ERROR] cli_csv 缺少列: {args.cli_id_col}"

    # 统一成同名键，方便 merge
    if args.img_id_col != "patient_id":
        img_df = img_df.rename(columns={args.img_id_col: "patient_id"})
    if args.cli_id_col != "patient_id":
        cli_df = cli_df.rename(columns={args.cli_id_col: "patient_id"})

    # 处理“同名列冲突”：你两边都有 label
    # 以图像表的 label 作为最终训练标签（保持图像划分一致）
    if "label" in img_df.columns:
        img_df = img_df.rename(columns={"label": "label_img"})

    if "label" in cli_df.columns:
        if args.keep_clinical_label:
            cli_df = cli_df.rename(columns={"label": "clinical_label"})
        else:
            cli_df = cli_df.drop(columns=["label"])

    # 体征表 patient_id 必须唯一（否则会把图像行膨胀）
    dup = cli_df["patient_id"].duplicated().sum()
    if dup > 0:
        # 保守处理：保留第一次出现
        print(f"[WARN] 体征表 patient_id 有重复 {dup} 条，将保留第一次出现的记录。")
        cli_df = cli_df.drop_duplicates(subset=["patient_id"], keep="first")

    # 左连接：以图像表为主，保证图像的 split 完全不变
    merged = img_df.merge(cli_df, on="patient_id", how="left")

    # 输出统计
    n_img = len(img_df)
    n_merge = len(merged)
    n_missing_cli = merged.isna().any(axis=1).sum()

    print("=" * 60)
    print("[INFO] Merge Finished")
    print(f"Image rows:  {n_img}")
    print(f"Merged rows: {n_merge}")
    print(f"Rows with ANY missing clinical fields (may include non-clinical cols): {n_missing_cli}")
    print("=" * 60)

    # 保存：NaN 会写成空单元格（满足“空出来不删除”）
    ensure_dir(os.path.dirname(args.out_csv))
    merged.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved merged csv -> {args.out_csv}")


if __name__ == "__main__":
    main()

# python src/scripts/merge_image_and_clinical_to_splits.py --img_csv data/图像数据标签.csv --cli_csv data/体征数据.csv --out_csv data/splits_with_clinical.csv