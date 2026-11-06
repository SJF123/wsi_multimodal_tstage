import torch
from src.model import TStageModel,VARIANTS,total_loss

def test_every_model_variant():
    torch.manual_seed(4)
    p=torch.randn(3,5,384);t=torch.randn(3,768);labels=torch.randn(4,768);y=torch.tensor([0,1,2])
    for key in VARIANTS:
        model=TStageModel(key);o=model(p,t,labels)
        assert o['logits'].shape==(3,4),key
        loss,*_=total_loss(o,y,alignment=model.cfg.alignment)
        assert torch.isfinite(loss),key

def test_parameter_limits():
    model=TStageModel()
    assert abs(model.img_res.effective_beta()-.2)<1e-6
    assert 0<torch.sigmoid(model.fusion_phi)<1

def test_table2_alignment_d1_distinct():
    assert VARIANTS['alignment_linear'].fusion=='linear'
    assert VARIANTS['D1'].fusion=='weighted_mlp'
    assert VARIANTS['N1']==VARIANTS['D2']
