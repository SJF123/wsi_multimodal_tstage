import numpy as np
from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score,confusion_matrix

def scores(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int)
    return {'accuracy':float(100*accuracy_score(y,p)),
            'macro_f1':float(100*f1_score(y,p,labels=[0,1,2,3],average='macro',zero_division=0)),
            'quadratic_kappa':float(cohen_kappa_score(y,p,labels=[0,1,2,3],weights='quadratic')),
            'stage_mae':float(np.abs(y-p).mean()),
            'gt1_stage_error':float(100*np.mean(np.abs(y-p)>1))}

def summarize(predictions):
    import pandas as pd
    df=pd.read_csv(predictions)
    rows=[]
    for (model,seed),g in df.groupby(['model','seed']):
        rows.append({'model':model,'seed':seed,**scores(g.y_true,g.y_pred)})
    d=pd.DataFrame(rows)
    agg=d.groupby('model')[['accuracy','macro_f1','quadratic_kappa','stage_mae','gt1_stage_error']].agg(['mean','std'])
    return d,agg

def paired_bootstrap(all_predictions,model='ours',baseline='late_fusion',reps=10000,seed=2026):
    """Bootstrap patients (same IDs in both methods) and average per-seed metric deltas; percentile CI."""
    import pandas as pd
    df=pd.read_csv(all_predictions)
    one=df[df.model==model];two=df[df.model==baseline]
    shared=sorted(set(one.seed)&set(two.seed))
    if len(shared)!=5: raise ValueError(f'Need all five matched seeds for {model}/{baseline}: got {shared}')
    aligned=[]
    for s in shared:
        a=one[one.seed==s][['patient_id','y_true','y_pred']].rename(columns={'y_pred':'ours_pred'})
        b=two[two.seed==s][['patient_id','y_true','y_pred']].rename(columns={'y_pred':'base_pred'})
        m=a.merge(b,on=['patient_id','y_true'],validate='one_to_one')
        if len(m)!=len(a) or len(m)!=len(b):raise ValueError('Mismatched patient IDs or ground truth')
        aligned.append(m.sort_values('patient_id').reset_index(drop=True))
    ids=aligned[0].patient_id.tolist()
    if any(d.patient_id.tolist()!=ids for d in aligned):raise ValueError('Different patient sets between runs')
    y=np.stack([g.y_true.values for g in aligned]);p=np.stack([g.ours_pred.values for g in aligned]);q=np.stack([g.base_pred.values for g in aligned]);n=len(ids)
    rng=np.random.default_rng(seed)
    def delta(ix):
        return np.mean([np.array([scores(y[k,ix],p[k,ix])[v]-scores(y[k,ix],q[k,ix])[v] for v in ('accuracy','macro_f1')]) for k in range(5)],axis=0)
    point=delta(np.arange(n));boot=np.array([delta(rng.integers(0,n,n)) for _ in range(reps)])
    lo,hi=np.percentile(boot,[2.5,97.5],axis=0)
    return {'comparison':f'{model} - {baseline}','delta_acc_pp':point[0], 'accuracy_ci_lo':lo[0], 'accuracy_ci_hi':hi[0],
            'delta_f1_pp':point[1], 'f1_ci_lo':lo[1], 'f1_ci_hi':hi[1], 'bootstrap_replicates':reps}
