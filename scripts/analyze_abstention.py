"""Describe one predeclared 0.05-distance abstention rule over five seed artifacts.

2024 results are *exploratory* and must never be used to optimize thresholds.
All seeds must have matching dates, labels, horizon and dataset SHA-256.
"""
from __future__ import annotations
import argparse,io,json,zipfile
from pathlib import Path
import numpy as np
from src.abstention import evaluate_policy,DEFAULT_DISTANCE

def analyze(seed_archives:dict[int,Path], *, distance=DEFAULT_DISTANCE):
    if set(seed_archives) != set(range(5)):
        raise ValueError('Exactly seeds 0-4 are required.')
    rows=[];canonical=None;canonical_fwd=None
    for seed in sorted(seed_archives):
        with zipfile.ZipFile(seed_archives[seed]) as z:
            metrics_names=[n for n in z.namelist() if n.endswith('/metrics.json')]
            preds_names=[n for n in z.namelist() if n.endswith('/preds.npz')]
            if len(metrics_names)!=1 or len(preds_names)!=1:raise ValueError(f'Incomplete archive for seed {seed}')
            meta=json.loads(z.read(metrics_names[0]))
            if int(meta['config']['seed']) != seed:raise ValueError(f'Wrong seed in archive {seed}')
            with np.load(io.BytesIO(z.read(preds_names[0])), allow_pickle=False) as d:
                required=['te','t_idx','y','fwd','p_lstm_ohlc']
                if any(k not in d for k in required): raise ValueError(f'Missing prediction fields in seed {seed}')
                te=d['te'].copy();t_idx=d['t_idx'].copy();y=d['y'].copy();fwd=d['fwd'].copy();p=d['p_lstm_ohlc'].copy()
            # Hash arrays to catch swapped markets / test intervals.
            import hashlib
            digest=hashlib.sha256(np.ascontiguousarray(np.stack([te,t_idx,y],axis=0)).tobytes()).hexdigest()
            key=(meta['data_provenance']['sha256'],meta['config']['horizon'],digest,len(y))
            if canonical is None:
                canonical=key;canonical_fwd=fwd
            elif key!=canonical or not np.allclose(fwd,canonical_fwd,atol=1e-12,rtol=0):
                raise ValueError('Seed datasets, targets or test periods do not match')
            if len(y)!=meta['n_test']:raise ValueError('Metric sample count differs from prediction file')
            row=evaluate_policy(p,y,fwd,horizon=meta['config']['horizon'],distance=distance,cost=meta['config']['cost'])
            row['seed']=seed;row['unfiltered_accuracy']=float(((p>=.5)==y).mean())
            rows.append(row)
    avg={}
    for k in ['unfiltered_accuracy','coverage','selected_direction_accuracy',
              'nonoverlap_active_fraction','net_log_pnl_proxy']:
        vals=[r[k] for r in rows if r[k] is not None]
        avg[k+'_mean']=float(np.mean(vals)) if vals else None
        avg[k+'_seed_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
    return {
        'dataset_sha256':canonical[0],'horizon_candles':int(canonical[1]),
        'test_observations_per_seed':canonical[3],
        'policy':'Fixed absolute distance >= 0.05 from probability 0.5, otherwise flat; short exposure allowed.',
        'interpretation':'2024 retrospective exploratory diagnostic ONLY; not a validated strategy or probability calibration.',
        'seeds':rows,'aggregate':avg}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-dir',default='.')
    p.add_argument('--out',default='results_2024_abstention.json')
    p.add_argument('--distance',type=float,default=DEFAULT_DISTANCE)
    a=p.parse_args()
    files={i:Path(a.archive_dir)/f'btcusdt_2024_seed{i}.zip' for i in range(5)}
    result=analyze(files,distance=a.distance)
    Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result['aggregate'],indent=2))
    print('Wrote',a.out)
if __name__=='__main__':main()
