from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[1]))
"""Train frozen-encoder image-only head from patch vectors; each encoder has its OWN manifest with matching dimensionality."""
import argparse,json,random
from pathlib import Path
import torch,numpy as np,pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.data import PatientDataset,collate_patient
from src.image_baselines import ImageBaseline,GatedMILBaseline
from src.metrics import scores

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--name',required=True)
    p.add_argument('--pooling',choices=['mean','attention','gated_mil'],default='mean')
    p.add_argument('--image-dim',type=int,default=384);p.add_argument('--output',default='outputs');p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55]);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=.001);p.add_argument('--patience',type=int,default=8);p.add_argument('--patch-seed',type=int,default=2026)
    a=p.parse_args()
    if a.name.lower()=='clam':
        raise ValueError('The bundled gated_mil example is NOT full CLAM. Use the verified official CLAM training implementation and record its provenance.')
    if a.name.lower() in ('uni','conch','uni_attention','conch_attention') and a.image_dim==384:
        print('VERIFY: use dimensions/features of the actual official UNI/CONCH checkpoint; 384 is just a CLI default.')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for seed in a.seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        loaders={sp:DataLoader(PatientDataset(a.manifest,sp,seed=a.patch_seed,feature_dim=a.image_dim),batch_size=a.batch_size,shuffle=sp=='train',collate_fn=collate_patient) for sp in ['train','val','test']}
        model=(GatedMILBaseline(a.image_dim) if a.pooling=='gated_mil' else ImageBaseline(a.image_dim,a.pooling)).to(device)
        optim=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
        base=Path(a.output)/a.name/str(seed);base.mkdir(parents=True,exist_ok=True)
        best=-1;bad=0;history=[]
        def evaluate(split,ep):
            model.eval();rows=[];losses=[]
            with torch.no_grad():
                for b in loaders[split]:
                    o=model(b['patches'].to(device),b['mask'].to(device));y=b['target'].to(device)
                    losses.append(float(F.cross_entropy(o['logits'],y)))
                    prob=o['logits'].softmax(-1).cpu().numpy()
                    for i,pid in enumerate(b['patient_id']):
                        rows.append({'model':a.name,'seed':seed,'epoch':ep,'patient_id':pid,'y_true':int(y[i]),'y_pred':int(prob[i].argmax()),
                                     **{f'prob_T{k+1}':prob[i,k] for k in range(4)},'embedding':json.dumps(o['fused'][i].cpu().numpy().tolist())})
            df=pd.DataFrame(rows);return scores(df.y_true,df.y_pred),df,float(np.mean(losses))
        for ep in range(1,a.epochs+1):
            model.train();ls=[]
            for b in loaders['train']:
                optim.zero_grad();o=model(b['patches'].to(device),b['mask'].to(device));loss=F.cross_entropy(o['logits'],b['target'].to(device));loss.backward();optim.step();ls.append(float(loss))
            va,_,vl=evaluate('val',ep);tr,_,_=evaluate('train',ep)
            history.append({'epoch':ep,'train_acc':tr['accuracy'],'train_f1':tr['macro_f1'],'train_loss':np.mean(ls),'val_acc':va['accuracy'],'val_f1':va['macro_f1'],'val_loss':vl})
            if va['macro_f1']>best+1e-9:
                best=va['macro_f1'];bad=0;best_ep=ep;torch.save(model.state_dict(),base/'best.pt')
            else:bad+=1
            if a.patience and bad>=a.patience:break
        pd.DataFrame(history).to_csv(base/'history.csv',index=False)
        model.load_state_dict(torch.load(base/'best.pt',map_location=device,weights_only=True))
        for split in ['train','val','test']:
            met,preds,_=evaluate(split,best_ep);preds.to_csv(base/f'{split}_predictions.csv',index=False)
            if split=='test':(base/'summary.json').write_text(json.dumps({'model':a.name,'seed':seed,'best_epoch':best_ep,'test':met},indent=2))
if __name__=='__main__':main()
