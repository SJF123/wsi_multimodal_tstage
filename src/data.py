"""Patient-level features; manifest CSV contains patient_id, split, stage, patch_feature_path, clinical_embedding_path, clinical_text."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLINICAL_COLUMNS=['age_at_index','age_at_diagnosis','gender','race','ethnicity','year_of_birth','year_of_diagnosis','primary_diagnosis','icd_10_code','tissue_or_organ_of_origin']
STAGES={'T1':0,'T2':1,'T3':2,'T4':3}
LABEL_DESCRIPTIONS=[f'The patient\'s T stage is T{k}.' for k in range(1,5)]


def clinical_text(row):
    parts=[]
    for col in CLINICAL_COLUMNS:
        v=row.get(col)
        if pd.notna(v) and str(v).strip() and str(v).strip().lower() not in ('nan','none'):
            parts.append(f'{col}: {v}')
    return ' ; '.join(parts) if parts else 'unknown'


def load_feature(path):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f'Missing feature file: {p}')
    if p.suffix.lower()=='.npy': a=np.load(p,allow_pickle=False)
    elif p.suffix.lower()=='.npz':
        with np.load(p,allow_pickle=False) as z: a=z['features'] if 'features' in z else z[z.files[0]]
    elif p.suffix.lower() in ('.pt','.pth'):
        a=torch.load(p,map_location='cpu',weights_only=True)
        if isinstance(a,dict): a=a['features'] if 'features' in a else a['embeddings']
        if isinstance(a,torch.Tensor): a=a.numpy()
    else: raise ValueError(f'Unsupported feature format: {p}')
    return np.asarray(a,dtype=np.float32)


class PatientDataset(Dataset):
    def __init__(self,manifest,split,n_patches=200,seed=42,replace_short=True,feature_dim=384):
        df=pd.read_csv(manifest) if not isinstance(manifest,pd.DataFrame) else manifest.copy()
        self.df=df[df['split']==split].reset_index(drop=True)
        if self.df.empty: raise ValueError(f'No data in split={split}')
        self.n=n_patches; self.seed=int(seed);self.replace_short=replace_short;self.feature_dim=feature_dim
        missing={'patient_id','stage','patch_feature_path','clinical_embedding_path'}-set(self.df.columns)
        if missing: raise ValueError(f'Missing manifest columns: {missing}')

    def __len__(self):return len(self.df)

    def __getitem__(self,index):
        r=self.df.iloc[index]
        patches=load_feature(r.patch_feature_path)
        if patches.ndim!=2 or patches.shape[1]!=self.feature_dim or len(patches)==0:
            raise ValueError(f'Expected nonempty [N,{self.feature_dim}] patch array: {r.patch_feature_path}, got {patches.shape}')
        # Sampling is deterministic per patient + preprocessing seed, preventing hidden run-to-run changes.
        import hashlib
        key=int.from_bytes(hashlib.sha256(str(r.patient_id).encode()).digest()[:4],'little')
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,key]))
        if len(patches)>=self.n:
            ix=rng.choice(len(patches),size=self.n,replace=False)
        elif self.replace_short:
            ix=rng.choice(len(patches),size=self.n,replace=True)
        else: ix=np.arange(len(patches))
        feat=patches[ix]
        text=load_feature(r.clinical_embedding_path).reshape(-1)
        if text.shape!=(768,):raise ValueError(f'Expected [768] clinical embedding, got {text.shape}')
        stage=str(r.stage).strip().upper()
        if stage in STAGES: y=STAGES[stage]
        elif stage in ('0','1','2','3'):y=int(stage)
        else:raise ValueError(f'Bad stage for patient {r.patient_id}: {r.stage}')
        return {'patient_id':str(r.patient_id),'patches':torch.tensor(feat),'mask':torch.ones(len(feat),dtype=torch.bool),
                'clinical':torch.tensor(text),'target':torch.tensor(y,dtype=torch.long)}


def collate_patient(batch):
    n=max(len(b['patches']) for b in batch)
    patches=torch.zeros(len(batch),n,batch[0]['patches'].shape[-1])
    mask=torch.zeros(len(batch),n,dtype=torch.bool)
    for i,b in enumerate(batch):
        s=len(b['patches']);patches[i,:s]=b['patches'];mask[i,:s]=True
    return {'patient_id':[b['patient_id'] for b in batch], 'patches':patches,'mask':mask,
            'clinical':torch.stack([b['clinical'] for b in batch]),'target':torch.stack([b['target'] for b in batch])}


def validate_manifest(csv_file,require_counts=False):
    d=pd.read_csv(csv_file)
    assert d.patient_id.nunique()==len(d),'Duplicate patient_id; combine slide features at patient level first'
    assert set(d.split).issubset({'train','val','test'})
    if require_counts:
        assert d.split.value_counts().to_dict()=={'train':477,'val':61,'test':67},'Patient counts differ from manuscript'
        assert [sum((d.split=='test') & (d.stage.astype(str).str.upper()==f'T{i}')) for i in range(1,5)]==[12,16,25,14]
    return d
