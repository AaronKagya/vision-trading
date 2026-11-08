"""Build the offline BTCUSDT 2025-Q1 historical position-replay demo.

Input provenance is verified against the archived GitHub research artifacts.
Do not interpret the result as a brokerage backtest or a live paper account.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from src.paper_replay import ensure_matches_archived_policy

TEMPLATE=Path(__file__).resolve().parents[1]/'site'/'paper_template.html'


def member(zf,suffix):
    names=[n for n in zf.namelist() if n.endswith(suffix) and not n.startswith('__MACOSX/')]
    if len(names)!=1:raise ValueError(f'Expected one {suffix}, got {len(names)}')
    return zf.read(names[0])


def build_payload(holdout_path,epoch_selection_path,epoch_holdout_path):
    with zipfile.ZipFile(holdout_path) as z:
        met=json.loads(member(z,'/metrics.json'))
        with np.load(io.BytesIO(member(z,'/preds.npz')),allow_pickle=False) as x:
            for key in ('p_lstm_ohlc','y','fwd','t_idx'):
                if key not in x:raise ValueError('Missing source '+key)
            p=x['p_lstm_ohlc'].astype(float)
            y=x['y'].astype(int)
            r=x['fwd'].astype(float)
            t=x['t_idx'].astype(int)
    n=len(p)
    if (n!=met['n_test'] or n<8000 or y.shape!=(n,) or r.shape!=(n,) or t.shape!=(n,)
        or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or not np.isfinite(r).all()
        or np.any(p<0) or np.any(p>1) or np.any(np.diff(t)!=1)):
        raise ValueError('Invalid original holdout prediction alignment')
    if (not met['training_must_not_use_2025'] or
        met['external_test_period']!='2025-01-01 through 2025-03-31, 15-minute candles' or
        met.get('seed')!=0 or met.get('horizon')!=4):
        raise ValueError('Holdout is not the locked external Q1 research run')
    start=np.datetime64('2024-01-01T00:00:00')
    times=start+t.astype('timedelta64[m]')*15
    if str(times[0])[:10]!='2025-01-01' or str(times[-1])[:10]!='2025-03-31':
        raise ValueError('Unexpected holdout decision timestamps')
    archived=met['models']['lstm_ohlc']['abstain_policy']
    v=ensure_matches_archived_policy(p,y,r,archived)
    with zipfile.ZipFile(epoch_selection_path) as z:
        selection=json.loads(member(z,'selected.json'))
    with zipfile.ZipFile(epoch_holdout_path) as z:
        q2=json.loads(member(z,'/holdout.json'))
    if (selection.get('seeds')!=[0,1,2,3,4] or selection.get('selected_max_epochs')!=20 or
       q2.get('status')!='external_test_completed' or
       q2.get('validation_budget_selection')!=selection['selected_max_epochs'] or
       q2.get('data_sha256_2024_matched')!=selection['data_sha256_2024'] or
       q2.get('holdout_period')!='2025-04-01 to 2025-06-30, UTC' or
       q2.get('n_overlap_forecasts',0)<8000):
        raise ValueError('Phase 4 provenance / five-seed validation mismatch')
    rows=[]
    pos=v['positions']
    for i in range(len(pos)):
        j=i*4
        rows.append([str(times[j]).replace('T',' ')[:16],
                     round(float(v['probabilities'][i]),7),int(pos[i]),
                     round(float(v['forward_log_returns'][i]),9),
                     round(float(v['gross'][i]),9),
                     round(float(v['costs'][i]),9),
                     round(float(v['net'][i]),9),
                     round(float(v['cumulative_net'][i]),9)])
    return {
        'schema':1,'instrument':'BTCUSDT (Binance spot)','timeframe':'15-minute candles',
        'forecast_horizon_minutes':60,'replay_window':'2025-01-01 through 2025-03-31 (UTC)',
        'rows':rows,'summary':v['summary'],
        'q1':{'accuracy':met['models']['lstm_ohlc']['accuracy'],
              'majority_accuracy':met['models']['majority']['accuracy'],
              'selected_direction_accuracy':archived['selected_direction_accuracy']},
        'q2':{'selected_max_epochs':selection['selected_max_epochs'],
              'lstm_accuracy_mean':q2['mean_lstm_accuracy'],
              'majority_accuracy':q2['majority_training_only']['accuracy'],
              'seeds':list(q2['models'])},
        'sources':{
            'q1_holdout_sha256':hashlib.sha256(Path(holdout_path).read_bytes()).hexdigest(),
            'phase4_selection_sha256':hashlib.sha256(Path(epoch_selection_path).read_bytes()).hexdigest(),
            'phase4_holdout_sha256':hashlib.sha256(Path(epoch_holdout_path).read_bytes()).hexdigest(),
        },
        'limitations':'Fixed historical 2025-Q1 signals. Disjoint one-hour position proxy, no actual fills, market spread, slippage, funding, leverage, liquidation, API orders, or verified live profitability.'
    }


def build(holdout_path,epoch_selection_path,epoch_holdout_path,out):
    payload=build_payload(holdout_path,epoch_selection_path,epoch_holdout_path)
    templ=TEMPLATE.read_text(encoding='utf-8')
    if templ.count('@@DATA_JSON@@')!=1:raise RuntimeError('Template must have exactly one data slot')
    js=json.dumps(payload,ensure_ascii=True,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    destination=Path(out);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(templ.replace('@@DATA_JSON@@',js),encoding='utf-8')
    print('Built',destination,'from',len(payload['rows']),'non-overlapping historical predictions')
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--q1-holdout-zip',required=True)
    p.add_argument('--epoch-selection-zip',required=True)
    p.add_argument('--epoch-holdout-zip',required=True)
    p.add_argument('--out',default='site/paper/index.html')
    a=p.parse_args()
    build(a.q1_holdout_zip,a.epoch_selection_zip,a.epoch_holdout_zip,a.out)

if __name__=='__main__':main()
