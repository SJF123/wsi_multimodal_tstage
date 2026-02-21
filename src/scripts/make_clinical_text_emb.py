# 离线生成 DistilBERT 临床文本向量
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

CLINICAL_CSV = "data/clinical_patient_expanded_labeled.csv"
OUT_PT = "data/clinical_text_emb_distilbert.pt"

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128

TEXT_COLS = [
    "primary_diagnosis", "morphology", "icd_10_code",
    "tissue_or_organ_of_origin", "tumor_grade", "classification_of_tumor"
]

def norm_pid(x: str) -> str:
    return str(x).strip().upper()

def build_text(row):
    parts = []
    for c in TEXT_COLS:
        if c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            continue
        v = str(v).strip()
        if v and v.lower() not in ["nan", "none"]:
            parts.append(f"{c}: {v}")
    return " ; ".join(parts) if parts else "unknown"

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_csv(CLINICAL_CSV)
    df["patient_id"] = df["patient_id"].apply(norm_pid)
    df["text"] = df.apply(build_text, axis=1)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    emb_dict = {}
    for _, r in tqdm(df.iterrows(), total=len(df)):
        pid = r["patient_id"]
        text = r["text"]

        batch = tok(
            text, truncation=True, padding=True, max_length=MAX_LEN,
            return_tensors="pt"
        ).to(device)

        out = model(**batch)  # last_hidden_state: [1, L, H]
        # mean pooling (简洁稳定)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu()  # [768]
        emb_dict[pid] = emb

    torch.save(emb_dict, OUT_PT)
    print("Saved:", OUT_PT)
    print("Patients:", len(emb_dict))
    print("One emb shape:", next(iter(emb_dict.values())).shape)

if __name__ == "__main__":
    main()
