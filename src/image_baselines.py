"""Image-only baselines using external frozen patch-feature files.
MeanPoolingClassifier for ResNet50/DINO; AttentionClassifier for UNI/CONCH;
CLAMBaseline is a CLAM-style attention MIL approximation (not official CLAM implementation).
"""
import torch
from torch import nn
from torch.nn import functional as F
from .model import AttentionPooling

class ImageBaseline(nn.Module):
    def __init__(self,input_dim,pooling='mean',dropout=.1):
        super().__init__();self.pooling=pooling
        self.pool=AttentionPooling(input_dim,128,dropout) if pooling=='attention' else None
        self.classifier=nn.Sequential(nn.Linear(input_dim,256),nn.ReLU(),nn.Dropout(dropout),nn.Linear(256,4))
    def forward(self,patches,mask=None):
        if self.pool is not None: feature,weights=self.pool(patches,mask)
        else:
            feature=patches.mean(1) if mask is None else (patches*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            weights=None
        return {'logits':self.classifier(feature),'fused':feature,'weights':weights}

class GatedMILBaseline(nn.Module):
    """CLAM-like gated MIL aggregator only. Do not call official CLAM: lacks instance clustering loss."""
    def __init__(self,input_dim,hidden=256):
        super().__init__();self.tanh=nn.Linear(input_dim,hidden);self.sigmoid=nn.Linear(input_dim,hidden)
        self.att=nn.Linear(hidden,1);self.classifier=nn.Linear(input_dim,4)
    def forward(self,patches,mask=None):
        scores=self.att(torch.tanh(self.tanh(patches))*torch.sigmoid(self.sigmoid(patches))).squeeze(-1)
        if mask is not None:scores=scores.masked_fill(~mask,-1e9)
        weights=scores.softmax(1);fused=(weights.unsqueeze(-1)*patches).sum(1)
        return {'logits':self.classifier(fused),'fused':fused,'weights':weights}
