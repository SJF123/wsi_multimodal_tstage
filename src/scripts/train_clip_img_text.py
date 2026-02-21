import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.datasets.clip_img_text_dataset import CLIPImgTextDataset

# 路径（一般不改）
SPLITS_CSV = "data/splits_labeled.csv"
TEXT_EMB_PT = "data/clinical_text_emb_distilbert.pt"
FEAT_DIR = "features/vit_patch_feats"

NUM_CLASSES = 4
MAX_PATCHES = 200

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_CLIP = 0.2
TEMPERATURE = 0.07

# --------- encoders ---------
class ImageEncoder(nn.Module):
    def __init__(self, in_dim=384, emb_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, patch_feats):
        # patch_feats: [B,N,D] 采用平均
        img_vec = patch_feats.mean(dim=1)          # [B,D]
        #img_vec = patch_feats
        z = self.proj(img_vec)                     # [B,emb]
        z = F.normalize(z, dim=-1)
        return z, img_vec

class TextEncoder(nn.Module):
    def __init__(self, in_dim=768, emb_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t):
        z = self.proj(t)
        z = F.normalize(z, dim=-1)
        return z

class FusionClassifier(nn.Module):
    def __init__(self, img_dim=384, txt_emb_dim=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim + txt_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, img_vec, txt_emb):
        h = torch.cat([img_vec, txt_emb], dim=1)
        return self.net(h)

class CLIPImgText(nn.Module):
    def __init__(self, img_in=384, txt_in=768, emb_dim=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.img = ImageEncoder(img_in, emb_dim)
        self.txt = TextEncoder(txt_in, emb_dim)
        self.cls = FusionClassifier(img_dim=img_in, txt_emb_dim=emb_dim, num_classes=num_classes)

    def forward(self, patch_feats, text_vec):
        z_img, img_vec = self.img(patch_feats)
        z_txt = self.txt(text_vec)
        logits = self.cls(img_vec, z_txt)
        return logits, z_img, z_txt

def clip_loss(z_img, z_txt, temperature=0.07):
    logits = (z_img @ z_txt.t()) / temperature  # [B,B]
    labels = torch.arange(z_img.size(0), device=z_img.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for patch, t, y in loader:
            patch = patch.to(device)
            t = t.to(device)
            y = y.to(device)
            logits, _, _ = model(patch, t)
            pred = logits.argmax(dim=1).cpu().tolist()
            ys += y.cpu().tolist()
            ps += pred
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = CLIPImgTextDataset(SPLITS_CSV, TEXT_EMB_PT, FEAT_DIR, split="train", max_patches=MAX_PATCHES)
    val_ds   = CLIPImgTextDataset(SPLITS_CSV, TEXT_EMB_PT, FEAT_DIR, split="val",   max_patches=MAX_PATCHES)
    test_ds  = CLIPImgTextDataset(SPLITS_CSV, TEXT_EMB_PT, FEAT_DIR, split="test",  max_patches=MAX_PATCHES)

    print("Samples Train/Val/Test:", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model = CLIPImgText().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    best_state, best_val_f1 = None, -1

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for patch, t, y in train_loader:
            patch = patch.to(device)
            t = t.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits, z_img, z_txt = model(patch, t)

            loss_ce = ce(logits, y)
            loss_cl = clip_loss(z_img, z_txt, TEMPERATURE)
            loss = loss_ce + LAMBDA_CLIP * loss_cl

            loss.backward()
            opt.step()
            total += loss.item()

        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | loss={total/len(train_loader):.4f} | val_acc={val_acc:.4f} | val_macroF1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    print("\n=== CLIP-style Image + DistilBERT(Text) Fusion (best on val macro-F1) ===")
    print(f"Test acc={test_acc:.4f} | Test macro-F1={test_f1:.4f}")

if __name__ == "__main__":
    main()
