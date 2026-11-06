"""Run validation-only model capacity audit on full, exact BTCUSDT 2024 history."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd
from src.data import load_candles
from src.epoch_audit import (make_audit_data,fit_audit,save_checkpoints,
    BUDGETS,WINDOW,HORIZON,PATIENCE,EMBARGO,LR,BS)


def run(data,out,seed):
    if seed<0:raise ValueError('Seed must be nonnegative')
    p=Path(data)
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    df=load_candles(data)
    num,labels,tr,va,idx=make_audit_data(df)
    hist,shots,sd=fit_audit(num,labels,tr,va,seed=seed,budgets=BUDGETS)
    config={'model':'lstm_ohlc','seed':seed,'budgets':list(BUDGETS),'max_epochs':max(BUDGETS),
            'patience':PATIENCE,'batch_size':BS,'learning_rate':LR,'window':WINDOW,
            'horizon':HORIZON,'embargo':EMBARGO,'train_last_utc':df.time.iloc[idx[tr[-1]]].isoformat(),
            'validation_first_utc':df.time.iloc[idx[va[0]]].isoformat(),
            'validation_last_utc':df.time.iloc[idx[va[-1]]].isoformat()}
    path=Path(out);path.mkdir(parents=True,exist_ok=True)
    result={'seed':seed,'data_sha256_2024':digest,'data_rows':int(len(df)),
            'n_train':int(len(tr)),'n_validation':int(len(va)),'config':config,
            'epochs_executed':len(hist),'stopped_early':False,
            'old_patience_early_stop_would_trigger_at':next((h['epoch'] for h in hist if h['old_patience_stopping_triggered']),None),
            'history':hist,'budgets':{str(b):{'selected_epoch':v['selected_epoch'],
                                  'validation_bce':v['best_validation_bce']} for b,v in shots.items()},
            'selection_rule':'Choose epoch cap with lowest mean validation BCE across all five seeds; ties choose smaller cap; never use 2025 labels.'}
    (path/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    pd.DataFrame(hist).to_csv(path/'history.csv',index=False)
    save_checkpoints(path,shots,sd,config=config,data_sha256=digest)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/BTCUSDT_15m.csv')
    p.add_argument('--out',default='results_epoch_seed_0')
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    x=run(a.data,a.out,a.seed)
    print('Training audit summary:',x['seed'],x['budgets'],flush=True)

if __name__=='__main__':main()
