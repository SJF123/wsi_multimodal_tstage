from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[1]))
"""WSI -> patient-level patch RGB arrays; configure actual magnification/thresholds in CLI.
Requires openslide-python, openslide shared library, Pillow, pandas, numpy.
"""
import argparse,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

def extract(csv_path,out_dir,patch_size=224,level=0,rgb_mean_max=235,min_variance=80,
            n_patches=200,seed=2026,short_policy='replacement'):
    import openslide
    d=pd.read_csv(csv_path);needed={'patient_id','slide_path'}-set(d)
    if needed:raise ValueError(f'Need CSV columns {needed}')
    outdir=Path(out_dir);outdir.mkdir(parents=True,exist_ok=True);rows=[]
    for pid,group in d.groupby('patient_id'):
        candidates=[]
        for path in group.slide_path:
            with openslide.OpenSlide(str(path)) as slide:
                w,h=slide.level_dimensions[level];downsample=slide.level_downsamples[level]
                for yy in range(0,h-patch_size+1,patch_size):
                    for xx in range(0,w-patch_size+1,patch_size):
                        img=slide.read_region((int(xx*downsample),int(yy*downsample)),level,(patch_size,patch_size)).convert('RGB')
                        arr=np.asarray(img)
                        if float(arr.mean())>rgb_mean_max or float(arr.var())<min_variance:continue
                        candidates.append((path,xx,yy,level))
        if len(candidates)==0:raise ValueError(f'No valid tissue patches for patient {pid}')
        hash_seed=int.from_bytes(hashlib.sha256(str(pid).encode()).digest()[:4],'little')
        rng=np.random.default_rng(np.random.SeedSequence([seed,hash_seed]))
        if len(candidates)>=n_patches:ix=rng.choice(len(candidates),n_patches,replace=False)
        elif short_policy=='replacement':ix=rng.choice(len(candidates),n_patches,replace=True)
        elif short_policy=='keep':ix=np.arange(len(candidates))
        else:raise ValueError(f'Too few patches for {pid}: {len(candidates)}')
        patient_dir=outdir/str(pid);patient_dir.mkdir(exist_ok=True)
        for j,k in enumerate(ix):
            path,xx,yy,ll=candidates[int(k)]
            with openslide.OpenSlide(str(path)) as slide:
                downsample=slide.level_downsamples[ll]
                slide.read_region((int(xx*downsample),int(yy*downsample)),ll,(patch_size,patch_size)).convert('RGB').save(patient_dir/f'{j:03d}.png')
        rows.append({'patient_id':pid,'patch_dir':str(patient_dir.resolve()),'eligible_candidates':len(candidates),'selected':len(ix)})
    pd.DataFrame(rows).to_csv(outdir/'patch_manifest.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--slides',required=True);p.add_argument('--output',required=True)
    p.add_argument('--level',type=int,default=0);p.add_argument('--patch-size',type=int,default=224)
    p.add_argument('--rgb-mean-max',type=float,default=235);p.add_argument('--min-variance',type=float,default=80)
    p.add_argument('--n-patches',type=int,default=200);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--short-policy',choices=['replacement','keep','error'],default='replacement')
    a=p.parse_args();extract(a.slides,a.output,a.patch_size,a.level,a.rgb_mean_max,a.min_variance,a.n_patches,a.seed,a.short_policy)
