from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import pandas as pd
from src.data import CLINICAL_COLUMNS,clinical_text

def main():
    p=argparse.ArgumentParser();p.add_argument('--patients',required=True,help='CSV patient_id,stage,split and clinical fields')
    p.add_argument('--feature-manifest',required=True,help='CSV patient_id,patch_feature_path')
    p.add_argument('--output',default='data/manifest_before_text_embeddings.csv')
    a=p.parse_args();pat=pd.read_csv(a.patients);feat=pd.read_csv(a.feature_manifest)
    d=pat.merge(feat[['patient_id','patch_feature_path']],on='patient_id',how='inner',validate='one_to_one')
    if len(d)!=len(pat):raise ValueError('Some patients lack extracted patch features')
    d['clinical_text']=d.apply(clinical_text,axis=1)
    o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);d.to_csv(o,index=False)
if __name__=='__main__':main()
