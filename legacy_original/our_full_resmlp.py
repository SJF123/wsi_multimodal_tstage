import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.residual_mlp import ResidualMLP

class AttentionPooling(nn.Module):
    """
    Attention pooling for patch features.
    Input : patch_feats [B, N, D]
    Output: pooled      [B, D]
    """
    def __init__(self, dim, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)   # score for each patch
        )

    def forward(self, x):
        # x: [B,N,D]
        scores = self.net(x).squeeze(-1)      # [B,N]
        weights = torch.softmax(scores, dim=1)  # [B,N]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B,D]
        return pooled, weights


class OurFullResMLP(nn.Module):
    """
    Our-Full + ResidualMLP + Attention Pooling
    - patch_feats: [B,N,384] -> attn_pool -> [B,384] -> ResidualMLP -> proj -> z_img [B,emb_dim]
    - cli_emb:     [B,768]   -> ResidualMLP -> proj -> z_cli [B,emb_dim]
    - fusion: concat([img_ref(384), cli_f(emb_dim)]) -> fusion_mlp -> f_fusion [B,emb_dim]
    - classifier: f_fusion -> logits [B,C]
    """

    def __init__(
        self,
        img_dim=384,
        cli_dim=768,
        emb_dim=256,
        num_classes=4,
        alpha_img=0.2,
        alpha_cli=0.2,
        learnable_alpha=False,
        dropout=0.1,
        attn_hidden=128,
    ):
        super().__init__()

        # 0) Attention pooling (new)
        self.attn_pool = AttentionPooling(img_dim, hidden=attn_hidden, dropout=dropout)

        # 1) Residual refinement for each branch (as your paper)
        self.img_ft = ResidualMLP(
            img_dim, hidden=512, dropout=dropout,
            alpha=alpha_img, learnable_alpha=learnable_alpha
        )
        self.cli_ft = ResidualMLP(
            cli_dim, hidden=1024, dropout=dropout,
            alpha=alpha_cli, learnable_alpha=learnable_alpha
        )

        # 2) Projections to shared space (for CLIP loss)
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.cli_proj = nn.Sequential(
            nn.Linear(cli_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # 3) Clinical projection for fusion
        self.cli_fusion_proj = nn.Sequential(
            nn.Linear(cli_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )

        # 4) Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )

        # 5) Classifier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, patch_feats, cli_emb, return_attn=False):
        """
        patch_feats: [B,N,384]
        cli_emb    : [B,768]
        """
        # A) attention pool -> [B,384]
        img_vec, attn_w = self.attn_pool(patch_feats)  # img_vec [B,384], attn_w [B,N]

        # B) residual refinement
        img_ref = self.img_ft(img_vec)    # [B,384]
        cli_ref = self.cli_ft(cli_emb)    # [B,768]

        # C) CLIP embeddings (2D!)
        z_img = F.normalize(self.img_proj(img_ref), dim=-1)  # [B,emb_dim]
        z_cli = F.normalize(self.cli_proj(cli_ref), dim=-1)  # [B,emb_dim]

        # D) fusion feature
        cli_f = self.cli_fusion_proj(cli_ref)  # [B,emb_dim]
        f_fusion = F.normalize(self.fusion(torch.cat([img_ref, cli_f], dim=1)), dim=-1)  # [B,emb_dim]

        logits = self.classifier(f_fusion)  # [B,C]

        if return_attn:
            return logits, z_img, z_cli, f_fusion, attn_w
        return logits, z_img, z_cli, f_fusion
