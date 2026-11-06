"""Regression tests for reviewer-audit corrections."""
import json
import numpy as np
import pandas as pd
import torch
from src.model import TStageModel
from src.report import _embeddings,draw_tsne,report


def test_a1_preserve_residual_magnitude():
    model=TStageModel('ours').eval()
    x=torch.randn(2,5,384);t=torch.randn(2,768)
    with torch.no_grad():
        ri,rt,_=model.features(x,t)
        pooled,_=model.pool(x)
        zi=torch.nn.functional.normalize(model.img_proj(pooled),dim=-1)
        zt=torch.nn.functional.normalize(model.cli_proj(t),dim=-1)
        assert torch.allclose(ri,model.img_res(zi),atol=1e-6)
        assert torch.allclose(rt,model.cli_res(zt),atol=1e-6)


def test_a2_n2_fusion_alpha_gradient():
    model=TStageModel('N2');labels=torch.randn(4,768)
    out=model(torch.randn(3,7,384),torch.randn(3,768),labels)
    out['logits'].sum().backward()
    assert model.fusion_phi.grad is not None
    assert model.fusion_phi.grad.abs().item()>0


def test_table2_alignment_baseline_does_not_use_n2_alpha():
    model=TStageModel('alignment_linear');labels=torch.randn(4,768)
    out=model(torch.randn(3,7,384),torch.randn(3,768),labels)
    out['logits'].sum().backward()
    assert model.fusion_phi.grad is None


def test_report_no_embeddings_and_figures(tmp_path):
    root=tmp_path/'out';seed=root/'tfidf'/'11';seed.mkdir(parents=True)
    d=pd.DataFrame({'patient_id':[f'p{i}' for i in range(8)],'y_true':[0,1,2,3,0,1,2,3],
                    'y_pred':[0,1,2,3,0,2,2,3]})
    assert _embeddings(d) is None
    d.to_csv(seed/'test_predictions.csv',index=False)
    report(root,baseline_names=(),reps=20)
    assert (root/'figures'/'tfidf'/'cm_11.png').exists()
    assert not (root/'figures'/'tfidf'/'tsne_11.png').exists()
    assert (root/'metrics_mean_sd.csv').exists()


def test_tsne_predicted_labels(tmp_path):
    d=pd.DataFrame({'patient_id':[str(i) for i in range(15)],
                    'y_true':np.arange(15)%4,'y_pred':(np.arange(15)+1)%4,
                    'embedding':[json.dumps([float(i),float(i%5),float(i%3),float(i%7)]) for i in range(15)]})
    assert draw_tsne(d,tmp_path/'tsne.png',11)
    assert (tmp_path/'tsne.png').exists()
