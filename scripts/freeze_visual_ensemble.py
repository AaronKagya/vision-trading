"""Fail-closed archival of all 5 frozen 2024 visual world-model checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, stdev
import zipfile

from scripts.train_2024_visual_shadow import (BATCH_SIZE,EPOCHS,EXPECTED_DATA_SHA256,
                                               IMAGE_SIZE,K,LAMBDA,LR,SEEDS,WINDOW)


def freeze(root: str | Path, out: str | Path):
    base=Path(root)
    dest=Path(out)
    dest.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in SEEDS:
        folders=list((base/f'visual-seed-{seed}').rglob('visual_training.json'))
        if len(folders)!=1:
            raise ValueError(f'Missing or duplicate visual training metadata for seed {seed}')
        meta=json.loads(folders[0].read_text(encoding='utf-8'))
        ck=folders[0].parent/f'wm_seed{seed}.pt'
        config=meta['config']
        for field,expected in [('seed',seed),('K',K),('window',WINDOW),('img',IMAGE_SIZE),
                               ('epochs_max',EPOCHS),('batch_size',BATCH_SIZE),('lr',LR),
                               ('lam',LAMBDA),('architecture','WorldModel'),('mode','wm'),('encoder','cnn')]:
            if config.get(field)!=expected:
                raise ValueError(f'seed {seed}: unexpected {field}')
        if meta['seed']!=seed or meta['dataset_sha256']!=EXPECTED_DATA_SHA256 or meta['data_rows']!=35136:
            raise ValueError(f'seed {seed}: dataset identity or seed mismatch')
        if meta['train_samples'] < 1000 or meta['validation_samples'] < 1000:
            raise ValueError(f'seed {seed}: inadequate sample counts')
        if not ck.is_file() or hashlib.sha256(ck.read_bytes()).hexdigest()!=meta['checkpoint_sha256']:
            raise ValueError(f'seed {seed}: missing or corrupted model weights')
        val=float(meta['validation_bce'])
        if not math.isfinite(val) or val<=0:
            raise ValueError(f'seed {seed}: invalid validation loss')
        records.append((meta,ck))
    sha={str(r['seed']):r['checkpoint_sha256'] for r,_ in records}
    manifest={'schema_version':1,'status':'five_seed_2024_only','model_id':'frozen_2024_visual_world_model_cnn_gru_five_seed_equal_mean_v1',
              'architecture':'WorldModel','mode':'wm','encoder':'cnn','window':WINDOW,'img':IMAGE_SIZE,
              'K':K,'training_data_sha256':EXPECTED_DATA_SHA256,'training_period':'2024-01 through 2024-09',
              'validation_period':'2024-10 through 2024-12',
              'forecast_horizon_candles':4,'ensemble_policy':'unweighted mean of five frozen seed probabilities',
              'seeds':list(SEEDS),'weights_sha256_by_seed':sha,
              'validation_bce_mean':mean(m['validation_bce'] for m,_ in records),
              'validation_bce_sd':stdev(m['validation_bce'] for m,_ in records),
              'limitations':['Chart imagery is deterministic from the same OHLC candles, not independent information.',
                             'Validation BCE is not proof of future accuracy or profitability.',
                             'Only prospectively paired, timestamp-identical predictions can support the next experiment.']}
    arc=dest/'frozen_2024_visual_ensemble.zip'
    with zipfile.ZipFile(arc,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for meta,ck in records:
            z.write(ck,f'wm_seed{meta["seed"]}.pt')
        z.writestr('visual_manifest.json',json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    (dest/'visual_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(f'Frozen all five visual checkpoints into {arc}. Mean 2024 validation BCE={manifest["validation_bce_mean"]:.6f}',flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='collected_visual')
    p.add_argument('--out',default='frozen_visual_ensemble')
    a=p.parse_args()
    freeze(a.root,a.out)

if __name__=='__main__':main()
