"""Patient-level CRC staging: projected features -> residual -> alignment/fusion -> label embeddings."""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


class AttentionPooling(nn.Module):
    """Legacy project's 384 -> 128 -> 1 attention architecture, with optional mask."""
    def __init__(self, dim=384, hidden=128, dropout=.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,hidden),nn.Tanh(),nn.Dropout(dropout),nn.Linear(hidden,1))

    def forward(self, x, mask=None):
        a=self.net(x).squeeze(-1)
        if mask is not None:
            if not torch.all(mask.any(dim=1)): raise ValueError('Each patient must have at least one patch')
            a=a.masked_fill(~mask,-1e9)
        a=F.softmax(a,dim=1)
        return (a.unsqueeze(-1)*x).sum(dim=1),a


class ResidualRefinement(nn.Module):
    def __init__(self, hidden, dropout=.1, init=.2, mode='residual'):
        super().__init__()
        if mode not in ('residual','plain','none'): raise ValueError(mode)
        self.mode=mode
        self.mlp=nn.Sequential(nn.Linear(256,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,256)) if mode!='none' else None
        if mode=='residual': self.theta=nn.Parameter(torch.tensor(math.log(init/(1-init)),dtype=torch.float32))

    def forward(self,z):
        if self.mode=='none': return z
        v=self.mlp(z)
        if self.mode=='plain': return v
        beta=torch.sigmoid(self.theta)
        return beta*v+(1-beta)*z

    def effective_beta(self):
        return float(torch.sigmoid(self.theta).detach()) if self.mode=='residual' else None


@dataclass
class ModelVariant:
    pooling:str='attention'
    image_res:str='residual'
    clinical_res:str='residual'
    alignment:bool=True
    fusion:str='weighted_mlp'   # weighted_mlp, concat_mlp, linear, gated
    prediction:str='label'       # label or classifier


VARIANTS={
    'ours':ModelVariant(), 'F0':ModelVariant(),
    'N1':ModelVariant(alignment=False),'D2':ModelVariant(alignment=False),
    'N2':ModelVariant(fusion='linear'),
    'N3':ModelVariant(image_res='none'),'N4':ModelVariant(clinical_res='none'),
    'N5':ModelVariant(pooling='mean'),
    'R1':ModelVariant(image_res='plain'),'R2':ModelVariant(clinical_res='plain'),
    'D1':ModelVariant(image_res='none',clinical_res='none'),
    'concat':ModelVariant(image_res='none',clinical_res='none',alignment=False,fusion='linear',prediction='classifier'),
    'late_fusion':ModelVariant(image_res='none',clinical_res='none',alignment=False,fusion='linear',prediction='classifier'),
    'gated_fusion':ModelVariant(image_res='none',clinical_res='none',alignment=False,fusion='gated',prediction='classifier'),
    'res_mlp_only':ModelVariant(alignment=False),
    # Table 2 specifically says contrastive + LINEAR; D1 retains MLP. Distinct.
    'alignment_linear':ModelVariant(image_res='none',clinical_res='none',fusion='linear'),
}


class TStageModel(nn.Module):
    """Frozen encoder outputs are inputs; projection head for label text is SHARED with clinical branch."""
    def __init__(self, variant='ours', image_dim=384, text_dim=768, scale=10., dropout=.1,
                 fusion_init=.5, single_modality=None):
        super().__init__()
        if variant not in VARIANTS: raise ValueError(f'Unknown variant {variant}')
        self.variant_name=variant
        self.cfg=VARIANTS[variant]
        self.single_modality=single_modality
        self.scale=float(scale)
        self.pool=AttentionPooling(image_dim,128,dropout)
        self.img_proj=nn.Sequential(nn.Linear(image_dim,256),nn.ReLU(),nn.Linear(256,256))
        self.cli_proj=nn.Sequential(nn.Linear(text_dim,256),nn.ReLU(),nn.Linear(256,256))
        self.img_res=ResidualRefinement(512,dropout,mode=self.cfg.image_res)
        self.cli_res=ResidualRefinement(1024,dropout,mode=self.cfg.clinical_res)
        self.fusion_phi=nn.Parameter(torch.tensor(math.log(fusion_init/(1-fusion_init)),dtype=torch.float32))
        self.fusion_head=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(dropout),nn.Linear(256,256))
        self.fusion_linear=nn.Linear(512,256)
        self.gate=nn.Linear(512,2)
        self.classifier=nn.Linear(256,4)
        self.img_classifier=nn.Linear(256,4)
        self.cli_classifier=nn.Linear(256,4)

    def features(self, patches, clinical, mask=None):
        if self.cfg.pooling=='mean':
            img = (patches*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1) if mask is not None else patches.mean(1)
            weights=None
        else: img, weights=self.pool(patches,mask)
        zi=F.normalize(self.img_proj(img),dim=-1)
        zt=F.normalize(self.cli_proj(clinical),dim=-1)
        ri=self.img_res(zi)
        rt=self.cli_res(zt)
        # Contrastive learning operates on refined embeddings (manuscript 10.30).
        # Do not silently add post-residual L2 normalization: equation (11) feeds
        # unnormalized refined features to fusion. Contrastive loss normalizes internally.
        return ri,rt,weights

    def forward(self, patches, clinical, label_text_embeddings=None, mask=None):
        ri,rt,w=self.features(patches,clinical,mask)
        if self.single_modality=='image':
            return {'logits':self.img_classifier(ri),'ri':ri,'rt':rt,'fused':ri,'weights':w}
        if self.single_modality=='clinical':
            return {'logits':self.cli_classifier(rt),'ri':ri,'rt':rt,'fused':rt,'weights':w}
        if self.variant_name=='late_fusion':
            logits=(self.img_classifier(ri)+self.cli_classifier(rt))/2
            return {'logits':logits,'ri':ri,'rt':rt,'fused':torch.cat([ri,rt],-1),'weights':w}
        joint=torch.cat([ri,rt],dim=-1)
        if self.cfg.fusion=='gated':
            gates=F.softmax(self.gate(joint),dim=-1)
            fused=self.fusion_head(torch.cat([gates[:,0:1]*ri,gates[:,1:2]*rt],-1))
        elif self.cfg.fusion=='linear':
            # N2 replaces only the MLP fusion head; retain the learned modality
            # weighting. Table-2 alignment_linear is a distinct unweighted baseline.
            if self.variant_name=='N2':
                alpha=torch.sigmoid(self.fusion_phi)
                joint=torch.cat([alpha*ri,(1-alpha)*rt],dim=-1)
            fused=self.fusion_linear(joint)
        elif self.cfg.fusion=='concat_mlp': fused=self.fusion_head(joint)
        else:
            alpha=torch.sigmoid(self.fusion_phi)
            fused=self.fusion_head(torch.cat([alpha*ri,(1-alpha)*rt],dim=-1))
        fused=F.normalize(fused,dim=-1)
        if self.cfg.prediction=='label':
            if label_text_embeddings is None: raise ValueError('Need four pretrained text embeddings [4,768] for label matching')
            # Reuse trained clinical projection; do NOT detach this projection operation.
            label_repr=F.normalize(self.cli_proj(label_text_embeddings),dim=-1)
            logits=self.scale*(fused@label_repr.T)
        else: logits=self.classifier(fused)
        return {'logits':logits,'ri':ri,'rt':rt,'fused':fused,'weights':w}


def symmetric_contrastive(ri,rt,temperature=.08):
    if ri.size(0)<2: return (ri.sum()+rt.sum())*0
    s=F.normalize(ri,dim=-1)@F.normalize(rt,dim=-1).T/temperature
    target=torch.arange(len(ri),device=ri.device)
    return (F.cross_entropy(s,target)+F.cross_entropy(s.T,target))/2


def total_loss(out,y,alignment=True,weight=.2,temperature=.08):
    # Align positive pairs at patient index, not patch index.
    stage=F.cross_entropy(out['logits'],y)
    align=symmetric_contrastive(out['ri'],out['rt'],temperature) if alignment else stage.new_zeros(())
    return stage+weight*align,stage,align
