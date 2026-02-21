import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

CLINICAL_CSV = "data/clinical_patient_expanded_labeled.csv"
SPLITS_LABELED = "data/splits_labeled.csv"

PATIENT_COL = "patient_id"
LABEL_COL = "label"   # 0..3
NUM_CLASSES = 4

# 你要用来构造“文本”的字段（存在则用，不存在会自动跳过）
TEXT_COLS_CAND = ["primary_diagnosis", "morphology", "icd_10_code", "tissue_or_organ_of_origin", "tumor_grade", "classification_of_tumor"]

MODEL_NAME = "distilbert-base-uncased"  # 轻量、够用；也可换 "bert-base-uncased"

def build_patient_split_map():
    df = pd.read_csv(SPLITS_LABELED)
    df[PATIENT_COL] = df[PATIENT_COL].astype(str).str.strip().str.upper()
    return df.groupby(PATIENT_COL)["split"].first().to_dict()

def build_text(row, text_cols):
    parts = []
    for c in text_cols:
        v = row.get(c, "")
        if pd.isna(v):
            v = ""
        v = str(v).strip()
        if v and v.lower() not in ["nan", "none"]:
            parts.append(f"{c}: {v}")
    return " ; ".join(parts) if parts else "unknown"

class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, text_col="text", label_col="y", max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, self.text_col]
        y = int(self.df.loc[idx, self.label_col])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_csv(CLINICAL_CSV)
    df[PATIENT_COL] = df[PATIENT_COL].astype(str).str.strip().str.upper()

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df[df[LABEL_COL].isin([0,1,2,3])].copy()
    df["y"] = df[LABEL_COL].astype(int)

    p2s = build_patient_split_map()
    df["split"] = df[PATIENT_COL].map(p2s)
    df = df.dropna(subset=["split"]).copy()

    text_cols = [c for c in TEXT_COLS_CAND if c in df.columns]
    print("Using text cols:", text_cols)
    df["text"] = df.apply(lambda r: build_text(r, text_cols), axis=1)

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(device)

    train_ds = SimpleTextDataset(train_df, tokenizer)
    val_ds   = SimpleTextDataset(val_df, tokenizer)
    test_ds  = SimpleTextDataset(test_df, tokenizer)

    args = TrainingArguments(
        output_dir="checkpoints/text_bert",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        num_train_epochs=6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=20,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(test_ds)
    print("\n=== Text-BERT Baseline (best on val macro-F1) ===")
    print(metrics)

if __name__ == "__main__":
    main()
