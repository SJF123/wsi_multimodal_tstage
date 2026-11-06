from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[1]))
"""Read beta_I/beta_T/alpha from the EXACT validation-selected model checkpoint."""
import argparse,json
from pathlib import Path
import pandas as pd
import torch
from src.model import TStageModel

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='outputs');p.add_argument('--model',default='ours')
    a=p.parse_args();root=Path(a.output)/a.model
    rows=[]
    for folder in sorted(root.iterdir()):
        ckfile=folder/'best.pt'
        if not ckfile.is_file():continue
        ck=torch.load(ckfile,map_location='cpu',weights_only=True)
        model=TStageModel(variant=ck['variant'],scale=ck.get('config',{}).get('scale',10.),single_modality=ck.get('config',{}).get('single_modality'))
        model.load_state_dict(ck['state_dict'])
        rows.append({'model':a.model,'seed':ck['seed'],'checkpoint_epoch':ck['epoch'],
                     'beta_I':model.img_res.effective_beta(),'beta_T':model.cli_res.effective_beta(),
                     'alpha':float(torch.sigmoid(model.fusion_phi).detach())})
    if not rows:raise FileNotFoundError(f'No best.pt files under {root}')
    df=pd.DataFrame(rows).sort_values('seed')
    df.to_csv(Path(a.output)/f'{a.model}_best_coefficients.csv',index=False)
    df[['beta_I','beta_T','alpha']].agg(['mean','std']).to_csv(Path(a.output)/f'{a.model}_coefficients_summary.csv')
    print(df.to_string(index=False))

if __name__=='__main__':main()
