"""Train one immutable 2024-only *visual world model* for prospective shadow evaluation.

Training: Jan--Sep 2024. Model selection: Oct--Dec 2024 validation BCE.
No 2025/2026 prices, labels or model-selection signals are read. A separate
five-seed aggregation step freezes the equal-mean ensemble. No orders.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from src.data import load_candles
from src.features import panels, render
from src import models as M

EXPECTED_DATA_SHA256 = '2cc72f14a30e03baa1aeff87d4d6c63b4359361146fa51b633b96a21e106de67'
K = 6
WINDOW = 32
IMAGE_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001
LAMBDA = 0.5
SEEDS = tuple(range(5))


def checked_split(df):
    """Reproduce Phase 4 fixed 2024 fit/val dates and label embargo, without test data."""
    start = pd.Timestamp('2024-01-01T00:00:00Z')
    val_start = pd.Timestamp('2024-10-01T00:00:00Z')
    end = pd.Timestamp('2025-01-01T00:00:00Z')
    if len(df) != 35136 or df.time.iloc[0] != start or df.time.iloc[-1] != end-pd.Timedelta(minutes=15):
        raise ValueError('Visual training requires exactly 2024 candles')
    if (df.time.diff().iloc[1:] != pd.Timedelta(minutes=15)).any():
        raise ValueError('Missing or irregular 2024 candles')
    panel_index, pan = panels(df, window=WINDOW)
    times = df.time.astype('int64').to_numpy()
    valid = np.arange(len(panel_index)-4, dtype=np.int64)
    train_candidates = valid[times[panel_index[valid]+4] < val_start.value]
    train = train_candidates[:-10]  # preserve original Phase 4 sample embargo
    train = train[train >= K-1]      # visual model sees the previous 5 chart frames
    val = valid[(times[panel_index[valid]] >= val_start.value)
                & (times[panel_index[valid]+4] < end.value)]
    labels = np.zeros(len(panel_index), dtype=np.int64)
    labels[valid] = (df.close.to_numpy()[panel_index[valid]+4] > df.close.to_numpy()[panel_index[valid]]).astype(np.int64)
    if not len(train) or not len(val) or len(train) < 1000 or len(val) < 1000:
        raise ValueError('Empty or inadequate 2024 split')
    if not (int(panel_index[train[-1]])+4 < int(panel_index[val[0]])):
        raise ValueError('Embargoed training/validation labels overlap')
    if not (times[panel_index[train]+4].max() < val_start.value):
        raise ValueError('Training labels would cross 2024-10-01')
    # Only train frames ending at or before t. Auxiliary t+1 is inside 2024 fit split.
    if times[panel_index[train[-1]] + 1] >= val_start.value:
        raise ValueError('Auxiliary target crosses train split')
    return pan, labels, train, val, panel_index


def train(data: str, out: str, seed: int, *, fit=M.fit_world_model):
    if seed not in SEEDS:
        raise ValueError('Only predetermined seeds 0..4 are allowed')
    source = Path(data)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != EXPECTED_DATA_SHA256:
        raise ValueError('Unexpected 2024 dataset identity; refusing to retrain on changed data')
    df = load_candles(source)
    pan, labels, tr, va, idx = checked_split(df)
    imgs = render(pan, size=IMAGE_SIZE)
    model, offs = fit(imgs, labels, tr, va, 'wm', K, 'cnn', EPOCHS, LR,
                      BATCH_SIZE, LAMBDA, seed, False)
    if not np.array_equal(offs, np.arange(1-K,1)):
        raise ValueError('Visual model produced unexpected or future frame offsets')
    val_bce = float(M._val_loss(model, imgs, labels, va, offs, BATCH_SIZE))
    if not math.isfinite(val_bce) or val_bce <= 0:
        raise ValueError('Nonfinite or invalid validation BCE')
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint = dest / f'wm_seed{seed}.pt'
    config = {'seed':seed, 'architecture':'WorldModel', 'mode':'wm', 'encoder':'cnn',
              'K':K, 'window':WINDOW, 'img':IMAGE_SIZE,'epochs_max':EPOCHS,
              'batch_size':BATCH_SIZE,'lr':LR,'lam':LAMBDA,
              'forecast_horizon_candles':4,
              'fit_period':'2024-01-01 through 2024-09-30, chronology-embargoed',
              'validation_period':'2024-10-01 through 2024-12-31',
              'selection':'validation-only early stopping, patience=3'}
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'model':'WorldModel','config':config,
                'data_sha256_2024':sha,'best_validation_bce':val_bce},checkpoint)
    test_pack = torch.load(checkpoint, map_location='cpu',weights_only=True)
    assert test_pack['model']=='WorldModel'
    metadata = {'schema_version':1,'seed':seed,'dataset_sha256':sha,
                'data_rows':len(df),'train_samples':len(tr),'validation_samples':len(va),
                'last_fit_candle_utc':df.time.iloc[idx[tr[-1]]].isoformat(),
                'first_validation_candle_utc':df.time.iloc[idx[va[0]]].isoformat(),
                'validation_bce':val_bce,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'checkpoint':checkpoint.name,'config':config,
                'scientific_note':'Validation loss is not prospective trading performance. No price prediction has been approved for trading.'}
    (dest/'visual_training.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(f'seed={seed} validation_BCE={val_bce:.6f} sha={metadata["checkpoint_sha256"]}',flush=True)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/BTCUSDT_15m.csv')
    parser.add_argument('--out',required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    train(args.data,args.out,args.seed)

if __name__=='__main__': main()
