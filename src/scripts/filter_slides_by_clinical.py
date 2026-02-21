import pandas as pd

# 读取 slides + clinical
slides = pd.read_csv("data/slides.csv")
clinical = pd.read_csv(r"E:/CRC-DL/02-生存预测/生存分析模型建立/data_cox/cleaned_data.csv")

# 规范化 patient_id（防止空格/大小写问题）
slides["patient_id_norm"] = slides["patient_id"].astype(str).str.strip().str.upper()
clinical_pids = set(
    clinical["case_submitter_id"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# 只保留 clinical 中存在的病人
matched = slides[slides["patient_id_norm"].isin(clinical_pids)].copy()
matched = matched.drop(columns=["patient_id_norm"])

matched.to_csv("data/slides_400_matched.csv", index=False)

print("Saved: data/slides_400_matched.csv")
print("Matched slides:", len(matched))
print("Matched patients:", matched["patient_id"].nunique())
