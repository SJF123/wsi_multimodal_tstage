"""Train patient-level variants with validation Macro-F1 checkpointing and per-patient predictions."""
import argparse,json,random,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data import PatientDataset,collate_patient,validate_manifest
from .model import TStageModel,total_loss
from .metrics import scores


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def run_eval(model,loader,label_embeddings,device,variant,split,seed,epoch):
    model.eval(); rows=[]; losses=[]
    with torch.no_grad():
        for b in loader:
            patches=b['patches'].to(device);cli=b['clinical'].to(device); mask=b['mask'].to(device);y=b['target'].to(device)
            out=model(patches,cli,label_embeddings,mask)
            l,stage,align=total_loss(out,y,alignment=model.cfg.alignment and model.single_modality is None)
            losses.append(l.item())
            probs=out['logits'].softmax(-1).cpu().numpy()
            emb=out['fused'].detach().cpu().numpy()
            for k,pid in enumerate(b['patient_id']):
                rows.append({'model':variant,'seed':seed,'epoch':epoch,'split':split,'patient_id':pid,'y_true':int(y[k]),
                             'y_pred':int(probs[k].argmax()),**{f'prob_T{i+1}':float(probs[k,i]) for i in range(4)},
                             'embedding':json.dumps(emb[k].tolist())})
    df=pd.DataFrame(rows);sc=scores(df.y_true,df.y_pred);sc['loss']=float(np.mean(losses))
    return sc,df


def train_one(manifest,label_file,outdir,variant='ours',seed=42,epochs=30,patience=8,batch_size=32,lr=.001,
              weight_decay=.0001,patch_seed=2026,device=None,scale=10.,single_modality=None,encoder_image_dim=384):
    """Read frozen feature tensors; no test access until checkpoint chosen using VAL Macro-F1."""
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    label=torch.load(label_file,map_location='cpu',weights_only=True)
    if isinstance(label,dict):label=label['embeddings']
    label=label.to(device).float()
    if label.shape!=(4,768):raise ValueError(f'Expected [4,768] label embeddings, got {label.shape}')
    # Preserve the exact manifest used by this run, without changing any IDs/labels.
    manifest_hash=hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    datasets={s:PatientDataset(manifest,s,seed=patch_seed,feature_dim=encoder_image_dim) for s in ('train','val','test')}
    loaders={s:DataLoader(ds,batch_size=batch_size,shuffle=s=='train',collate_fn=collate_patient,drop_last=False) for s,ds in datasets.items()}
    model=TStageModel(variant=variant,scale=scale,single_modality=single_modality,image_dim=encoder_image_dim).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=weight_decay)
    best=-float('inf');bad=0;history=[];best_epoch=None
    for epoch in range(1,epochs+1):
        model.train();ls=[]
        for b in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            out=model(b['patches'].to(device),b['clinical'].to(device),label,b['mask'].to(device))
            total,stage,align=total_loss(out,b['target'].to(device),alignment=model.cfg.alignment and model.single_modality is None)
            total.backward();optimizer.step();ls.append(float(total.detach().cpu()))
        tr,_=run_eval(model,loaders['train'],label,device,variant,'train',seed,epoch)
        va,_=run_eval(model,loaders['val'],label,device,variant,'val',seed,epoch)
        history.append({'epoch':epoch,'train_acc':tr['accuracy'],'train_f1':tr['macro_f1'],
                        'train_loss':float(np.mean(ls)),'val_acc':va['accuracy'],'val_f1':va['macro_f1'],'val_loss':va['loss']})
        print(f'{variant} seed {seed} epoch {epoch}: val Macro-F1 {va["macro_f1"]:.3f}')
        # Select solely on validation Macro-F1; deterministic tie-break earlier epoch.
        if va['macro_f1']>best+1e-9:
            best=va['macro_f1'];bad=0;best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'variant':variant,'seed':seed,'epoch':epoch,
                        'config':{'scale':scale,'single_modality':single_modality}},outdir/'best.pt')
        else:bad+=1
        if patience>0 and bad>=patience:break
    pd.DataFrame(history).to_csv(outdir/'history.csv',index=False)
    ck=torch.load(outdir/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(ck['state_dict'])
    result={}
    for split in ('train','val','test'):
        sc,df=run_eval(model,loaders[split],label,device,variant,split,seed,best_epoch)
        df.to_csv(outdir/f'{split}_predictions.csv',index=False)
        result[split]=sc
    result['seed']=seed;result['variant']=variant;result['best_epoch']=best_epoch
    result['manifest_sha256']=manifest_hash
    result['training_config']={'epochs_max':epochs,'patience':patience,'batch_size':batch_size,'lr':lr,'weight_decay':weight_decay,'patch_seed':patch_seed,'scale':scale,'checkpoint_selection':'validation_macro_f1','single_modality':single_modality}
    result['alpha']=float(torch.sigmoid(model.fusion_phi.detach()))
    result['beta_I']=model.img_res.effective_beta();result['beta_T']=model.cli_res.effective_beta()
    (outdir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (outdir/'best_coefficients.json').write_text(json.dumps({'seed':seed,'selected_epoch':best_epoch,'alpha':result['alpha'],'beta_I':result['beta_I'],'beta_T':result['beta_T']},indent=2))
    return result


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--manifest',required=True);a.add_argument('--label-embeddings',required=True)
    a.add_argument('--output',default='outputs');a.add_argument('--variants',nargs='+',default=['ours']);a.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55]);a.add_argument('--epochs',type=int,default=30);a.add_argument('--patience',type=int,default=8)
    a.add_argument('--batch-size',type=int,default=32);a.add_argument('--patch-seed',type=int,default=2026)
    a.add_argument('--lr',type=float,default=.001);a.add_argument('--weight-decay',type=float,default=.0001)
    a.add_argument('--require-paper-counts',action='store_true');a.add_argument('--image-dim',type=int,default=384)
    args=a.parse_args();validate_manifest(args.manifest,args.require_paper_counts)
    for variant in args.variants:
        for seed in args.seeds:
            single_modality='image' if variant=='image_only' else 'clinical' if variant=='clinical_only' else None
            model_variant='ours' if single_modality else variant
            train_one(args.manifest,args.label_embeddings,Path(args.output)/variant/str(seed),variant=model_variant,seed=seed,
                      epochs=args.epochs,patience=args.patience,batch_size=args.batch_size,patch_seed=args.patch_seed,
                      lr=args.lr,weight_decay=args.weight_decay,single_modality=single_modality,encoder_image_dim=args.image_dim)

if __name__=='__main__':main()
