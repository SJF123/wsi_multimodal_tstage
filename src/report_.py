"""Recompute results from real predictions; never derive outcomes from manuscript tables.

Outputs: per-seed / mean-SD metrics, patient-paired bootstrap, confusion matrices,
optional t-SNE (only where embeddings are present), per-configuration curves.
Figure 3's nine-panel composite is only generated from THREE VERIFIABLE histories;
they represent independently trained configurations, not internal joint branches.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.manifold import TSNE
from .metrics import summarize,paired_bootstrap

# Table 5 comparison IDs: use the actual output directory names.
# image_only (attention-based single-modality) is NOT dino_mean (mean pooling).
# The PFM comparator in the final Table 5 is CONCH + Attention Pooling.
DEFAULT_BASELINES=('dino_mean','late_fusion','gated_fusion','res_mlp_only','conch_attention')


def _embeddings(df):
    if 'embedding' not in df.columns or df['embedding'].isna().any():
        return None
    try:
        data=[np.asarray(json.loads(v),dtype=np.float32) for v in df['embedding']]
        if len(set(x.shape for x in data))!=1:return None
        result=np.stack(data)
        return result if result.ndim==2 and result.shape[1]>1 and np.isfinite(result).all() else None
    except (TypeError,ValueError,json.JSONDecodeError):
        return None


def draw_tsne(df,output,seed):
    """Mark errors with BLACK outlines, annotate their PREDICTED R0--R3 class."""
    embeddings=_embeddings(df)
    if embeddings is None:
        print(f'No compatible embeddings in {output}; skipping t-SNE (metrics still reported).')
        return False
    if len(df)<=3:return False
    xy=TSNE(n_components=2,random_state=seed,perplexity=min(15,len(df)-1),init='pca',learning_rate='auto').fit_transform(embeddings)
    fig,ax=plt.subplots(figsize=(7,6))
    truth=df.y_true.to_numpy(dtype=int)
    pred=df.y_pred.to_numpy(dtype=int)
    for k in range(4):
        selected=truth==k
        ax.scatter(xy[selected,0],xy[selected,1],label=f'T{k+1}',s=28,alpha=.85)
    wrong=np.flatnonzero(truth!=pred)
    if len(wrong):
        ax.scatter(xy[wrong,0],xy[wrong,1],facecolors='none',edgecolors='black',linewidths=1.4,s=88,zorder=5,label='Misclassified')
        for j in wrong:
            ax.annotate(f'R{pred[j]}',(xy[j,0],xy[j,1]),xytext=(4,3),textcoords='offset points',fontsize=8,color='black')
    ax.legend(fontsize=8)
    ax.set_title('Patient-level t-SNE (R0–R3: predicted stage of errors)')
    fig.tight_layout();fig.savefig(output,dpi=300);plt.close(fig)
    return True


def draw_histories(root,model,output=None):
    histories=[]
    for f in sorted((root/model).glob('*/history.csv')):
        d=pd.read_csv(f)
        if {'epoch','train_acc','val_acc','train_f1','val_f1','train_loss','val_loss'}.issubset(d.columns):
            histories.append(d.set_index('epoch'))
    if not histories:return False
    common=sorted(set.intersection(*[set(h.index) for h in histories]))
    if not common:return False
    fig,axes=plt.subplots(1,3,figsize=(16,4.7))
    for ax,(tr,va,title) in zip(axes,[('train_acc','val_acc','Accuracy'),('train_f1','val_f1','Macro-F1'),('train_loss','val_loss','Loss')]):
        for col,color,name in [(tr,'#B3D49B','Train'),(va,'#CDA3CB','Val')]:
            values=np.stack([h.loc[common,col].to_numpy(dtype=float) for h in histories])
            # Classification metrics are stored in percent; figure shows 0--1.
            if title!='Loss':values=values/100.0
            ax.plot(common,values.mean(0),color=color,marker='o',markersize=3,label=name)
            ax.fill_between(common,values.min(0),values.max(0),color=color,alpha=.17)
        ax.set_title(f'{model}: {title}');ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.3)
        if title!='Loss':ax.set_ylim(0,1)
    fig.suptitle(f'Independent {model} runs (n={len(histories)}; band=min–max)',fontsize=13)
    fig.tight_layout();out=output or root/'figures'/f'fig3_{model}.tiff';out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=300,pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
    return True


def report(root,baseline_names=DEFAULT_BASELINES,reps=10000):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    files=sorted(root.glob('*/*/test_predictions.csv'))
    if not files:raise ValueError(f'No test predictions under {root}; run real inference first')
    frames=[];summaries=[]
    for f in files:
        model=f.parent.parent.name;seed=int(f.parent.name)
        d=pd.read_csv(f)
        must={'patient_id','y_true','y_pred'}
        if not must.issubset(d.columns):raise ValueError(f'{f} lacks {sorted(must-set(d.columns))}')
        if d.patient_id.duplicated().any():raise ValueError(f'Duplicate patient IDs in {f}')
        d['model']=model;d['seed']=seed;frames.append(d)
        summary=f.parent/'summary.json'
        if summary.exists():
            j=json.loads(summary.read_text(encoding='utf-8'))
            summaries.append({'model':model,'seed':seed,'selected_epoch':j.get('best_epoch'),'alpha':j.get('alpha'),
                              'beta_I':j.get('beta_I'),'beta_T':j.get('beta_T')})
    allpred=pd.concat(frames,ignore_index=True)
    allpred.to_csv(root/'all_test_predictions.csv',index=False)
    per_seed,aggregate=summarize(root/'all_test_predictions.csv')
    per_seed.to_csv(root/'metrics_per_seed.csv',index=False)
    aggregate.to_csv(root/'metrics_mean_sd.csv')
    if summaries:
        coefficients=pd.DataFrame(summaries)
        coefficients.to_csv(root/'supplementary_coefficients.csv',index=False)
        beta=coefficients[coefficients.model.isin(['ours','F0'])].drop_duplicates(subset=['seed'])
        if not beta.empty:
            beta[['beta_I','beta_T']].agg(['mean','std']).to_csv(root/'supplementary_S2_mean_sd.csv')
    # Table 5: fail explicitly if any requested model or run is missing.
    # Never silently replace dino_mean with image_only or CONCH with UNI.
    available=set(allpred.model)
    required={'ours', *baseline_names}
    missing=sorted(required-available)
    if missing:
        raise ValueError(
            f'Table 5 cannot be computed: missing prediction directories {missing} '
            f'under {root}. Found: {sorted(available)}. '
            'Each required directory must contain <seed>/test_predictions.csv.'
        )

    reference=allpred[allpred.model=='ours']
    ref_seeds=set(reference.seed.unique())
    if not ref_seeds:
        raise ValueError('No runs found for ours.')
    for baseline in baseline_names:
        observed=set(allpred.loc[allpred.model==baseline,'seed'].unique())
        if observed != ref_seeds:
            raise ValueError(
                f'Table 5 seed mismatch for {baseline}: '
                f'ours={sorted(ref_seeds)}, baseline={sorted(observed)}'
            )
        for seed in sorted(ref_seeds):
            a=reference[reference.seed==seed].set_index('patient_id')['y_true']
            b=allpred[(allpred.model==baseline)&(allpred.seed==seed)].set_index('patient_id')['y_true']
            if set(a.index)!=set(b.index):
                raise ValueError(f'Table 5 patient-ID mismatch: {baseline}, seed={seed}')
            b=b.reindex(a.index)
            if not np.array_equal(a.to_numpy(),b.to_numpy()):
                raise ValueError(f'Table 5 ground-truth mismatch: {baseline}, seed={seed}')

    # Save the *actual* compared input files for review and audit.
    audit=[]
    for model in ['ours',*baseline_names]:
        for seed in sorted(ref_seeds):
            f=root/model/str(seed)/'test_predictions.csv'
            audit.append({'table':'Table 5','model':model,'seed':int(seed),
                          'prediction_file':str(f.resolve()),
                          'patients':int(len(allpred[(allpred.model==model)&(allpred.seed==seed)]))})
    pd.DataFrame(audit).to_csv(root/'table5_input_audit.csv',index=False)

    comparisons=[]
    for baseline in baseline_names:
        try:
            comparisons.append(paired_bootstrap(root/'all_test_predictions.csv','ours',baseline,reps=reps))
        except ValueError as exc:
            raise ValueError(f'Table 5 paired bootstrap failed for {baseline}: {exc}') from exc
    pd.DataFrame(comparisons).to_csv(root/'paired_bootstrap_table5.csv',index=False)
    for (model,seed),group in allpred.groupby(['model','seed']):
        outdir=root/'figures'/model;outdir.mkdir(parents=True,exist_ok=True)
        cm=confusion_matrix(group.y_true,group.y_pred,labels=[0,1,2,3])
        fig,ax=plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(cm,display_labels=['T1','T2','T3','T4']).plot(ax=ax,cmap='Blues',colorbar=False)
        ax.set_title(f'{model} | seed {seed}')
        fig.tight_layout();fig.savefig(outdir/f'cm_{seed}.png',dpi=300);plt.close(fig)
        draw_tsne(group,outdir/f'tsne_{seed}.png',seed)
    for model in sorted(set(allpred.model)):
        draw_histories(root,model)
    return per_seed,aggregate

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs');parser.add_argument('--bootstrap',type=int,default=10000)
    parser.add_argument('--baselines',nargs='+',default=list(DEFAULT_BASELINES),
                        help='Table 5 output-directory identifiers (default: dino_mean late_fusion gated_fusion res_mlp_only conch_attention)')
    args=parser.parse_args();report(args.output,baseline_names=tuple(args.baselines),reps=args.bootstrap)
