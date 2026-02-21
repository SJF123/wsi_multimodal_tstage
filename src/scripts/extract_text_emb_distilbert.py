import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/clinical_text_from_patient.csv")
    ap.add_argument("--text_col", default="clinical_text")
    ap.add_argument("--id_col", default="patient_id")

    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--batch_size", type=int, default=16)

    ap.add_argument("--out_pt", default="features/clinical_feats/clinical_text_emb_distilbert_patient.pt")
    ap.add_argument("--only_missing", action="store_true")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    for c in [args.text_col, args.id_col]:
        assert c in df.columns, f"[ERROR] CSV 缺少列: {c}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    # 已存在的 embedding（支持增量）
    if args.only_missing and os.path.exists(args.out_pt):
        text_emb = torch.load(args.out_pt, map_location="cpu")
        print(f"[INFO] Loaded existing embeddings: {len(text_emb)}")
    else:
        text_emb = {}

    ids = df[args.id_col].tolist()
    texts = df[args.text_col].fillna("").tolist()

    ensure_ids = []
    ensure_texts = []

    for pid, txt in zip(ids, texts):
        pid = str(pid)
        if args.only_missing and pid in text_emb:
            continue
        ensure_ids.append(pid)
        ensure_texts.append(txt)

    print(f"[INFO] To encode: {len(ensure_texts)} samples")

    for i in tqdm(range(0, len(ensure_texts), args.batch_size)):
        batch_text = ensure_texts[i:i + args.batch_size]
        batch_ids = ensure_ids[i:i + args.batch_size]

        enc = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        out = model(**enc)
        cls_emb = out.last_hidden_state[:, 0]  # CLS token
        cls_emb = cls_emb.cpu()

        for pid, emb in zip(batch_ids, cls_emb):
            text_emb[pid] = emb

    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    torch.save(text_emb, args.out_pt)

    print("=" * 60)
    print("[OK] Text embeddings saved")
    print("Output:", args.out_pt)
    print("Total embeddings:", len(text_emb))
    print("=" * 60)


if __name__ == "__main__":
    main()

# python src/scripts/extract_text_emb_distilbert.py --csv data/clinical_text_from_patient.csv --out_pt features/clinical_feats/clinical_text_emb_distilbert_patient.pt
