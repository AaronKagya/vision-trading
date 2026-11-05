"""Leakage-aware, reproducible walk-forward forecasts and research-only trading proxies.

Examples:
  python -m src.run_experiments --data synthetic_null --n-synth 1300 --epochs 1 --folds 2 --bootstrap 30
  python -m src.run_experiments --data data/BTCUSDT_15m.csv --epochs 10 --folds 4 --seed 0
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import load_candles, synthetic
from src.features import panels,render,numeric_features
from src import models as M
from src import metrics as X

BASE_MODELS=['majority','persistence','logreg_lags','lstm_ohlc','cnn_static','wm_nolatent','wm']
MODELS=BASE_MODELS+['fusion']

def build_folds(samples,folds=4,horizon=4,K=6):
    """Expanding folds with train -> embargo -> val -> embargo -> test.

    Train and validation label horizons do not overlap the following partition.
    Earlier test periods may enter training for later folds, as in live rolling retrains.
    """
    if folds<1:raise ValueError('folds must be >=1')
    n=len(samples)
    first_test=int(n*.5)
    edges=np.linspace(first_test,n,folds+1).astype(int)
    emb=horizon+K
    for fold in range(folds):
        start=int(edges[fold]);end=int(edges[fold+1]);tr_end=start-emb
        val_len=max(8,int(.15*tr_end))
        tr_stop=tr_end-val_len-emb
        if tr_stop<24 or end<=start:raise ValueError('Not enough observations for these folds/embargoes')
        tr=samples[:tr_stop];va=samples[tr_end-val_len:tr_end];te=samples[start:end]
        if not (tr.max()+horizon<va.min() and va.max()+horizon<te.min()):
            raise RuntimeError('Fold embargo violation')
        yield {'fold':fold,'tr':tr,'va':va,'te':te,'embargo':emb}

def calculate_metrics(P,y_test,forward_return,config,spread=None,bootstrap=1000):
    names=list(P)
    h=config['horizon'];cost=config['cost'];n=len(y_test)
    ppy=365*24*4/h # BTC 15m intervals; 24/7 illustrative annualization
    rets={m:X.strategy_returns(P[m],forward_return,cost,stride=h) for m in names}
    sr={m:float(r.mean()/(r.std(ddof=1)+1e-12)) for m,r in rets.items()}
    trial_srs=np.asarray(list(sr.values()))
    n_trials=len(names)+config['extra_trials']
    res={'config':config,'n_test':int(n),'up_rate_test':float(y_test.mean()),'models':{},
         'vs_lstm':{},'n_trials':n_trials,
         'metric_notes':{
             'total_return':'Legacy name: sum of linearized net log-PnL proxy, NOT simple equity return',
             'total_log_pnl_proxy':'Sum(pos*forward_log_price_change - per-unit turnover costs)',
             'implied_compounded_return':'expm1(total_log_pnl_proxy); illustrative proxy, not realized return',
             'n_trades':'Count of nonoverlapping evaluation intervals, not entry orders',
             'dsr':'Approximate deflated Sharpe probability; relies on an honest count of ALL tried configurations',
             'shorts':'Short exposure, execution, margin, slippage and funding are NOT modeled faithfully'}}
    for m in names:
        fm=X.forecast_metrics(P[m],y_test)
        ci=X.block_bootstrap(lambda idx,m=m:float(((P[m][idx]>=.5)==y_test[idx]).mean()),n,B=bootstrap,seed=0)
        total=float(rets[m].sum())
        res['models'][m]={**fm,'acc_ci':ci,'sharpe_net':X.sharpe(rets[m],ppy),
           'psr':X.psr(rets[m]),'dsr':X.deflated_sharpe(rets[m],trial_srs,n_trials),
           'total_return':total, # compatibility only; name is misleading
           'total_log_pnl_proxy':total,'implied_compounded_return':X.compounded_from_log_proxy(total),
           'n_trades':int(len(rets[m])),'n_trade_intervals':int(len(rets[m]))}
        if spread and m in spread:res['models'][m]['latent_spread']=float(np.mean(spread[m]))
    if 'lstm_ohlc' in names:
        for m in ['cnn_static','wm_nolatent','wm','fusion','persistence']:
            if m not in names:continue
            difference=lambda idx,m=m:float(((P[m][idx]>=.5)==y_test[idx]).mean()-((P['lstm_ohlc'][idx]>=.5)==y_test[idx]).mean())
            ci=X.block_bootstrap(difference,n,B=bootstrap,seed=1)
            res['vs_lstm'][m]={'acc_diff_vs_lstm':difference(np.arange(n)),'ci':ci,
                                'significant':bool(ci[0]>0 or ci[1]<0)}
    bh=float(forward_return[::h].sum())
    res['buy_hold_total_return']=bh # historical compatibility: LOG return
    res['buy_hold_log_return']=bh
    res['buy_hold_compounded_return']=float(np.expm1(bh))
    return res

def _sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def run(a):
    import torch
    start=time.time();out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    requested=(BASE_MODELS if a.models=='all' else MODELS if a.models=='phase2'
               else [x.strip() for x in a.models.split(',')])
    if not requested or len(set(requested))!=len(requested) or any(m not in MODELS for m in requested):
        raise ValueError(f'Invalid --models; valid choices: {MODELS}')
    if a.horizon<1 or a.K<1 or a.bs<2 or a.bootstrap<2:
        raise ValueError('horizon,K,bs must be positive, bootstrap>=2')
    M.ENC_OPTS['vit_res']=a.vit_res
    torch.manual_seed(a.seed)
    if a.data.startswith('synthetic'):
        kind='planted' if a.data=='synthetic_planted' else 'null'
        if a.data not in ('synthetic_null','synthetic_planted'):raise ValueError(a.data)
        df=synthetic(a.n_synth,kind,a.seed)
        sha=None
    else:
        df=load_candles(a.data)
        sha=_sha256(a.data)
    t_idx,pan=panels(df,a.window)
    imgs=render(pan,a.img)
    num=numeric_features(pan)
    close=df.close.to_numpy()[t_idx]
    h=a.horizon
    fwd=np.full(len(close),np.nan)
    fwd[:-h]=np.log(close[h:]/close[:-h])
    label=(fwd>0).astype(np.int64)
    raw_close=df.close.to_numpy()
    lag=np.stack([np.log(raw_close[t_idx-k]/raw_close[t_idx-k-1]) for k in range(5)],axis=1)
    momentum=np.zeros(len(close))
    momentum[h:]=np.log(close[h:]/close[:-h])
    samples=np.arange(a.K-1,len(close)-h)
    folds=list(build_folds(samples,a.folds,h,a.K))
    P={m:[] for m in requested};te_all=[];spreads={m:[] for m in requested if m in ('cnn_static','wm_nolatent','wm')}
    fold_info=[]
    for f in folds:
        fid=f['fold'];tr=f['tr'];va=f['va'];te=f['te']
        fold_info.append({'fold':fid,'train_n':int(len(tr)),'validation_n':int(len(va)),
                         'test_n':int(len(te)),'train_last_sample':int(tr[-1]),
                         'val_first_sample':int(va[0]),'val_last_sample':int(va[-1]),
                         'test_first_sample':int(te[0]),'embargo_samples':int(f['embargo'])})
        print(f'Fold {fid}: train {len(tr)} / val {len(va)} / test {len(te)}',flush=True)
        te_all.append(te)
        if 'majority' in P:
            P['majority'].append(np.full(len(te),float(label[tr].mean()>=.5)))
        if 'persistence' in P:
            P['persistence'].append((momentum[te]>0).astype(float))
        if 'logreg_lags' in P:
            reg=LogisticRegression(max_iter=500,random_state=a.seed)
            reg.fit(lag[tr],label[tr]);P['logreg_lags'].append(reg.predict_proba(lag[te])[:,1])
        if 'fusion' in P:
            cp=out/f'fusion_fold{fid}.pt' if a.save_checkpoints else None
            P['fusion'].append(M.fit_predict_fusion(imgs,num,label,tr,va,te,
                a.epochs,a.lr,a.img_bs or a.bs,a.seed,checkpoint=cp))
        if 'lstm_ohlc' in P:
            cp=out/f'lstm_fold{fid}.pt' if a.save_checkpoints else None
            P['lstm_ohlc'].append(M.fit_predict_lstm(num,label,tr,va,te,a.epochs,a.lr,a.bs,a.seed,checkpoint=cp))
        for name,mode,lmb in [('cnn_static','static',0.),('wm_nolatent','wm_nolatent',0.),('wm','wm',a.lam)]:
            if name not in P:continue
            model,offs=M.fit_world_model(imgs,label,tr,va,mode,a.K,a.encoder,a.epochs,
                a.lr,a.img_bs or a.bs,lmb,a.seed,a.pretrained)
            pred,spread=M.predict_world_model(model,offs,imgs,te,bs=max(a.img_bs or a.bs,8))
            P[name].append(pred);spreads[name].append(spread)
            if a.save_checkpoints:
                torch.save({'state_dict':model.cpu().state_dict(),
                    'encoder':a.encoder,'pretrained':a.pretrained,'K':a.K,
                    'mode':mode,'window':a.window,'img':a.img,'fold':fid},out/f'{name}_fold{fid}.pt')
            del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
    te=np.concatenate(te_all)
    probs={m:np.concatenate(values) for m,values in P.items()}
    config=vars(a).copy();config['models_evaluated']=requested
    config['phase2_candidate']='fusion' if 'fusion' in requested else None
    res=calculate_metrics(probs,label[te],fwd[te],config,spreads,a.bootstrap)
    res['data_provenance']={'sha256':sha,'rows':int(len(df)),
        'start_utc':df.time.iloc[0].isoformat(),'end_utc':df.time.iloc[-1].isoformat(),
        'synthetic':a.data.startswith('synthetic'),'source':'user-supplied file' if sha else 'generated simulation'}
    res['software']={'python':platform.python_version(),'numpy':np.__version__,
        'torch':torch.__version__,'gpu':str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')}
    res['folds']=fold_info
    res['elapsed_seconds']=round(time.time()-start,2)
    # Strict serialization: all reported quantities must be finite.
    (out/'metrics.json').write_text(json.dumps(res,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'preds.npz',te=te,y=label[te],fwd=fwd[te],
        t_idx=t_idx[te],**{f'p_{m}':v for m,v in probs.items()})
    np.save(out/'sample_imgs.npy',imgs[te][::max(len(te)//12,1)][:12])
    print(f'\nSaved {out}/metrics.json and preds.npz in {time.time()-start:.1f}s')
    for m in requested:
        r=res['models'][m]
        print(f'{m:14s}: acc {r["accuracy"]:.4f}, net Sharpe {r["sharpe_net"]:+.2f}, DSR {r["dsr"]:.3g}')
    return res

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='synthetic_planted')
    p.add_argument('--n-synth',type=int,default=7000)
    p.add_argument('--window',type=int,default=32)
    p.add_argument('--img',type=int,default=64)
    p.add_argument('--horizon',type=int,default=4)
    p.add_argument('--K',type=int,default=6)
    p.add_argument('--folds',type=int,default=4)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--bs',type=int,default=128)
    p.add_argument('--img-bs',type=int,default=None)
    p.add_argument('--vit-res',type=int,default=224)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--lam',type=float,default=.5)
    p.add_argument('--encoder',choices=['cnn','resnet18','vit_tiny'],default='cnn')
    p.add_argument('--pretrained',action='store_true')
    p.add_argument('--cost',type=float,default=.0005)
    p.add_argument('--extra-trials',type=int,default=0)
    p.add_argument('--models',default='all',help='all = original 7; phase2 = 7 plus fusion; or comma-separated names')
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--save-checkpoints',action='store_true')
    p.add_argument('--out',default='results')
    run(p.parse_args())

if __name__=='__main__':main()
