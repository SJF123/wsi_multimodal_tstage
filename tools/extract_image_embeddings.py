from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[1]))
"""Encode folders of RGB patches with pretrained timm encoders.
Use explicit model/checkpoint. DINO ViT-S/16 should produce 384 dimensions.
UNI/CONCH require model-specific authenticated weights/licenses from their official releases.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

class Patches(Dataset):
    def __init__(self,paths,size=224):
        self.paths=paths
        self.transform=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),
              transforms.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225))])
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):return self.transform(Image.open(self.paths[i]).convert('RGB'))

def extract(patch_manifest,model_name,out_dir,checkpoint=None,batch_size=32,device=None,image_size=224):
    import timm
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net=timm.create_model(model_name,pretrained=checkpoint is None,num_classes=0).to(device).eval()
    if checkpoint:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
        net.load_state_dict(state,strict=True)
    output=Path(out_dir);output.mkdir(parents=True,exist_ok=True);rows=[]
    d=pd.read_csv(patch_manifest)
    with torch.no_grad():
        for r in d.itertuples():
            imgs=sorted(Path(r.patch_dir).glob('*.png'))
            loader=DataLoader(Patches(imgs,image_size),batch_size=batch_size,shuffle=False)
            em=[]
            for batch in loader:
                z=net(batch.to(device))
                if z.ndim>2:z=z.flatten(1)
                em.append(z.cpu().numpy())
            result=np.concatenate(em,axis=0)
            dest=output/f'{r.patient_id}.npy';np.save(dest,result)
            rows.append({'patient_id':r.patient_id,'patch_feature_path':str(dest.resolve()),'feature_dim':result.shape[-1]})
    pd.DataFrame(rows).to_csv(output/'feature_manifest.csv',index=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--patch-manifest',required=True);a.add_argument('--model',required=True)
    a.add_argument('--output',required=True);a.add_argument('--checkpoint');a.add_argument('--image-size',type=int,default=224)
    x=a.parse_args();extract(x.patch_manifest,x.model,x.output,x.checkpoint,image_size=x.image_size)
