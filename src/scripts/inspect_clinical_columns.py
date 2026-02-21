import pandas as pd

CSV = "data/clinical_patient_400_matched.csv"
df = pd.read_csv(CSV)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# 猜测 label 列名
for cand in ["label", "t_stage", "T_stage", "y"]:
    if cand in df.columns:
        print("Label column:", cand)
        break

# 显示每列缺失比例（前20列）
na_ratio = df.isna().mean().sort_values(ascending=False)
print("\nTop missing ratio columns:")
print(na_ratio.head(20))
