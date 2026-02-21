import pandas as pd
import numpy as np

# 1. 读取“已匹配”的 slides
slides = pd.read_csv("data/slides_400_matched.csv")

# 2. 读取原始临床数据
clinical = pd.read_csv(
    r"E:/CRC-DL/02-生存预测/生存分析模型建立/data_cox/cleaned_data.csv"
)

# 3. 只保留 slides 中出现的病人
keep_pids = set(slides["patient_id"].astype(str))
clinical = clinical[
    clinical["case_submitter_id"].astype(str).isin(keep_pids)
].copy()

# 4. 选择最基础、最稳妥的一组临床特征（后面可以再加）
cols = []
for c in ["case_submitter_id", "ajcc_pathologic_t", "age_at_index", "gender", "race"]:
    if c in clinical.columns:
        cols.append(c)

df = clinical[cols].copy()

# 5. T 分期映射：T0–T4 → 0–4
def map_t(x):
    if pd.isna(x):
        return np.nan
    s = str(x).upper()
    if s.startswith("T0"):
        return 0
    if s.startswith("T1"):
        return 1
    if s.startswith("T2"):
        return 2
    if s.startswith("T3"):
        return 3
    if s.startswith("T4"):
        return 4
    return np.nan

df["label"] = df["ajcc_pathologic_t"].apply(map_t)

# 6. 聚合：每个病人只保留一行
def mode_or_nan(s):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    return s.value_counts().index[0]

agg = {
    "label": mode_or_nan,
}

if "age_at_index" in df.columns:
    agg["age_at_index"] = "median"
if "gender" in df.columns:
    agg["gender"] = mode_or_nan
if "race" in df.columns:
    agg["race"] = mode_or_nan

patient = (
    df.groupby("case_submitter_id", as_index=False)
      .agg(agg)
      .rename(columns={"case_submitter_id": "patient_id"})
)

# 7. 去掉没有 T 标签的病人
patient = patient.dropna(subset=["label"]).copy()
patient["label"] = patient["label"].astype(int)

# 8. 保存结果
patient.to_csv("data/clinical_patient_400_matched.csv", index=False)

print("Saved: data/clinical_patient_400_matched.csv")
print("Patients:", len(patient))
print("Label distribution:")
print(patient["label"].value_counts().sort_index())
