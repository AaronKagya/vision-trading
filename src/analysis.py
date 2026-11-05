"""Post-hoc descriptive robustness analysis. Not a hyperparameter selection tool.

python -m src.analysis --results results --data data/BTCUSDT_15m.csv
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_candles
from src.metrics import block_bootstrap

VISUAL=['cnn_static','wm_nolatent','wm']

def analyze(folder='results',data=None,bootstraps=300):
    folder=Path(folder)
    report=json.loads((folder/'metrics.json').read_text())
    x=np.load(folder/'preds.npz')
    y=x['y'];ids=x['t_idx']
    names=[k[2:] for k in x.files if k.startswith('p_')]
    preds={m:x['p_'+m] for m in names}
    out={'source':str(folder),'n_test':int(len(y)),
         'description':'Post-hoc descriptive only: no tuning or statistical claims across regimes.'}
    blocks=[];offset=0
    for fold in report['folds']:
        stop=offset+fold['test_n']
        blocks.append({'fold':int(fold['fold']), 'n':int(stop-offset),
                       'accuracy':{m:float(((preds[m][offset:stop]>=.5)==y[offset:stop]).mean()) for m in names}})
        offset=stop
    if offset!=len(y):raise ValueError('Fold sample counts do not match predictions.')
    out['by_test_fold']=blocks
    pairs={}
    for i,a in enumerate(VISUAL):
        for b in VISUAL[i+1:]:
            if a not in preds or b not in preds:continue
            ca=((preds[a]>=.5)==y).astype(float)
            cb=((preds[b]>=.5)==y).astype(float)
            d=ca-cb
            ci=block_bootstrap(lambda ix:float(d[ix].mean()),len(d),B=bootstraps,seed=7)
            pairs[f'{a}_minus_{b}']={'accuracy_difference':float(d.mean()),'ci95':ci}
    out['visual_comparisons']=pairs
    data_path=data or report['config'].get('data')
    if data_path and not str(data_path).startswith('synthetic') and Path(data_path).exists():
        df=load_candles(data_path)
        r=np.log(df.close).diff().fillna(0)
        vol=r.rolling(32,min_periods=32).std().to_numpy()[ids]
        # The volatility value for sample t includes only history through t.
        edges=np.quantile(vol,[1/3,2/3])
        regimes={}
        for label,mask in [('low',vol<=edges[0]),
                           ('medium',(vol>edges[0])&(vol<=edges[1])),
                           ('high',vol>edges[1])]:
            regimes[label]={'n':int(mask.sum()),
                'accuracy':{m:float(((preds[m][mask]>=.5)==y[mask]).mean()) for m in names}}
        out['volatility_terciles']=regimes
        out['volatility_note']='Cutoffs obtained retrospectively across the test period: descriptive only.'
    else:
        out['volatility_note']='Skipped: historical market CSV not accessible; pass --data.'
    (folder/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps(out,indent=2))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results');p.add_argument('--data',default=None)
    p.add_argument('--bootstrap',type=int,default=300)
    a=p.parse_args();analyze(a.results,a.data,a.bootstrap)
if __name__=='__main__':main()
