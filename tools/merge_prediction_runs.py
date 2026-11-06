from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[1]))
"""One CSV with model, seed, patient_id, y_true, y_pred for all completed runs."""
from src.report import report
import argparse
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');p.add_argument('--bootstrap',type=int,default=10000)
    a=p.parse_args();report(a.output,reps=a.bootstrap)
