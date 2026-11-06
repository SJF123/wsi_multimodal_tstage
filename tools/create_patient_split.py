from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[1]))
"""Create a NEW stratified patient split. This does NOT recreate historical paper split."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--patients',required=True,help='CSV with patient_id, stage')
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--train',type=int,default=477)
    p.add_argument('--val',type=int,default=61)
    p.add_argument('--test',type=int,default=67)
    args=p.parse_args();d=pd.read_csv(args.patients)
    if d.patient_id.duplicated().any():raise ValueError('One row per patient required')
    n=len(d)
    if args.train+args.val+args.test!=n:raise ValueError('Requested counts must match patient count')
    a,rest=train_test_split(d,test_size=args.val+args.test,random_state=args.seed,stratify=d.stage)
    b,c=train_test_split(rest,test_size=args.test,random_state=args.seed,stratify=rest.stage)
    for frame,split in [(a,'train'),(b,'val'),(c,'test')]:frame.loc[:,'split']=split
    out=pd.concat([a,b,c]).sort_values('patient_id')
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True);out.to_csv(target,index=False)
    print('WARNING: newly generated split is NOT the paper fixed split unless independently verified')
    print(pd.crosstab(out.split,out.stage))

if __name__=='__main__':main()
