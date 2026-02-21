import os
import argparse
import pandas as pd


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def build_clinical_text(row, ignore_cols):
    """
    将一行体征字段拼成自然语言文本
    缺失值自动跳过
    """
    texts = []

    for col, val in row.items():
        if col in ignore_cols:
            continue
        if pd.isna(val):
            continue

        texts.append(f"{col}: {val}")

    if len(texts) == 0:
        return ""

    return "; ".join(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/splits_with_clinical.csv")
    ap.add_argument("--out_csv", default="data/clinical_text_from_patient.csv")

    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--label_col", default="label_img")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # 必要字段检查
    for c in [args.patient_col, args.split_col, args.label_col]:
        assert c in df.columns, f"[ERROR] CSV 缺少必要列: {c}"

    # 不参与文本构建的列
    ignore_cols = {
        args.patient_col,
        args.split_col,
        args.label_col,
        "slide_id",
        "slide",
        "file_stem",
        "clinical_label",
    }

    records = []

    for _, row in df.iterrows():
        clinical_text = build_clinical_text(row, ignore_cols)

        records.append({
            "patient_id": row[args.patient_col],
            "split": row[args.split_col],
            "label": int(row[args.label_col]),
            "clinical_text": clinical_text
        })

    out_df = pd.DataFrame(records)

    ensure_dir(os.path.dirname(args.out_csv))
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("[OK] Clinical text CSV generated")
    print(f"Input : {args.csv}")
    print(f"Output: {args.out_csv}")
    print("Rows  :", len(out_df))
    print("=" * 60)


if __name__ == "__main__":
    main()


# python src/scripts/build_clinical_text_from_patient_csv.py --csv data/splits_with_clinical.csv --out_csv data/clinical_text_from_patient.csv