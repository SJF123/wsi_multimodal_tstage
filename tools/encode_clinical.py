from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[1]))
import argparse
from src.text_encoder import make_clinical_embeddings
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();make_clinical_embeddings(a.manifest,a.output)
