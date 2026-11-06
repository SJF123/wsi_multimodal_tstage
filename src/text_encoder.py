"""Frozen DistilBERT text embeddings, masked mean pooling, label descriptions same encoder."""
import torch
from transformers import AutoTokenizer,AutoModel
from .data import LABEL_DESCRIPTIONS,clinical_text,CLINICAL_COLUMNS,CLINICAL_COLUMNS

@torch.no_grad()
def encode_texts(texts,model_name='distilbert-base-uncased',batch_size=32,max_length=128,device=None):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    encoder=AutoModel.from_pretrained(model_name).to(device).eval()
    out=[]
    for st in range(0,len(texts),batch_size):
        token=tokenizer(texts[st:st+batch_size],return_tensors='pt',padding=True,truncation=True,max_length=max_length)
        token={k:v.to(device) for k,v in token.items()}
        z=encoder(**token).last_hidden_state
        mask=token['attention_mask'].unsqueeze(-1)
        out.append(((z*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu())
    return torch.cat(out) if out else torch.empty(0,768)


def make_clinical_embeddings(manifest_csv,output_csv,model_name='distilbert-base-uncased'):
    import pandas as pd
    from pathlib import Path
    d=pd.read_csv(manifest_csv)
    missing=set(CLINICAL_COLUMNS)-set(d.columns)
    if missing:
        raise ValueError(f'Strict clinical-text preparation requires all ten columns; missing: {sorted(missing)}')
    # SECURITY / LEAKAGE: never trust a pre-existing clinical_text column.
    if 'clinical_text' in d.columns:
        old=d['clinical_text'].fillna('unknown').astype(str)
        regenerated=d.apply(clinical_text,axis=1)
        changed=(old!=regenerated)
        print(f'Rebuilding clinical_text from approved fields; {int(changed.sum())} rows differ from supplied text.')
    d['clinical_text']=d.apply(clinical_text,axis=1)
    embeddings=encode_texts(d.clinical_text.fillna('unknown').tolist(),model_name)
    target=Path(output_csv);target.parent.mkdir(parents=True,exist_ok=True)
    embdir=target.parent/'clinical_embeddings';embdir.mkdir(exist_ok=True)
    paths=[]
    for p,z in zip(d.patient_id,embeddings):
        path=embdir/f'{p}.npy'
        import numpy as np
        np.save(path,z.numpy());paths.append(str(path.resolve()))
    d['clinical_embedding_path']=paths
    d.to_csv(target,index=False)
    audit=d[['patient_id','clinical_text']].copy()
    audit['num_fields_present']=d[CLINICAL_COLUMNS].notna().sum(axis=1)
    audit.to_csv(target.parent/'clinical_text_audit.csv',index=False)
    audit=d[['patient_id','clinical_text']].copy()
    audit['num_fields_present']=d[CLINICAL_COLUMNS].notna().sum(axis=1)
    audit.to_csv(target.parent/'clinical_text_audit.csv',index=False)
    labels=encode_texts(LABEL_DESCRIPTIONS,model_name)
    torch.save(labels,target.parent/'label_text_embeddings.pt')
    return target
