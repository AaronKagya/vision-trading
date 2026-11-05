"""Phase 2 out-of-sample diagnostics. Descriptive, NEVER a parameter tuner.

Run after an experiment:
  python -m src.phase2_eval --results results_phase2 --data data/BTCUSDT_15m.csv

All forecasts are test-block predictions. Foldwise volatility terciles are
classified using cutoffs learned from historical (pre-test) *prices only*.
There is NO threshold selection on test labels or test trading outcomes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from src.metrics import block_bootstrap
from src.data import load_candles


def _comparison(p, other, y, bootstraps=300, seed=17):
    a = ((p >= .5) == y).astype(float)
    b = ((other >= .5) == y).astype(float)
    diff = a - b
    ci = block_bootstrap(lambda ix: float(diff[ix].mean()), len(diff), B=bootstraps, seed=seed)
    return {'accuracy_delta_points': float(100 * diff.mean()),
            'ci95_points': [float(100 * ci[0]), float(100 * ci[1])],
            'method': 'paired moving-block bootstrap, descriptive post-hoc'}


def _confidence_bins(p, y):
    """Fixed intervals decided in advance. Do not optimize them on test data."""
    certainty = np.abs(p - .5)
    boundaries = (0., .02, .05, .10, .20, .5)
    rows = []
    for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (certainty >= lo) & ((certainty <= hi) if i == len(boundaries)-2 else (certainty < hi))
        n = int(mask.sum())
        rows.append({'distance_from_half_min': lo, 'distance_from_half_max': hi,
                     'observations': n, 'coverage': float(n / len(y)),
                     'accuracy': float(((p[mask] >= .5) == y[mask]).mean()) if n else None})
    return rows


def _historical_regimes(report, t_idx, prices, fold_info, predictions, y):
    """Train-calibrated volatility thresholds for test observations in each fold."""
    values = np.asarray(prices, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Invalid close prices')
    returns = np.diff(np.log(values), prepend=np.nan)
    lookback = 32
    # Volatility through each candle t, using only returns through t.
    sq = np.array([np.nanstd(returns[max(1, i-lookback+1):i+1], ddof=1)
                   if i >= lookback else np.nan for i in range(len(returns))])
    out = []
    offset = 0
    for f in fold_info:
        n = int(f['test_n'])
        end = offset + n
        if end > len(t_idx):
            raise ValueError('Fold boundaries exceed prediction count')
        # Training's last sample index is relative to the *panel array*;
        # t_idx maps that index to the corresponding raw market candle.
        from src.features import first_valid_index
        train_last_raw = int(f['train_last_sample']) + first_valid_index(report['config']['window'])
        available = sq[:train_last_raw+1]
        available = available[np.isfinite(available)]
        if len(available) < 30:
            raise ValueError('Insufficient pre-test history for volatility thresholds')
        low, high = np.quantile(available, [1/3, 2/3])
        test_vol = sq[t_idx[offset:end]]
        fold_rows = {'fold': int(f['fold']), 'thresholds_derived_from': 'historical train candles only',
                     'volatility_cutoffs': [float(low),float(high)], 'groups': {}}
        for name, mask in [('low', test_vol <= low),
                           ('medium', (test_vol > low) & (test_vol <= high)),
                           ('high', test_vol > high)]:
            n_mask = int(mask.sum())
            fold_rows['groups'][name] = {
                'observations': n_mask,
                'accuracy': {m: (float(((p[offset:end][mask] >= .5) == y[offset:end][mask]).mean())
                                 if n_mask else None) for m,p in predictions.items()}}
        out.append(fold_rows)
        offset = end
    if offset != len(t_idx):
        raise ValueError('Fold boundaries do not account for all predictions')
    return out


def evaluate(folder='results_phase2', data=None, bootstraps=300):
    folder = Path(folder)
    r = json.loads((folder/'metrics.json').read_text())
    with np.load(folder/'preds.npz', allow_pickle=False) as saved:
        y = saved['y'].astype(np.int64)
        t_idx = saved['t_idx'].astype(np.int64)
        probs = {key[2:]: np.asarray(saved[key], float) for key in saved.files if key.startswith('p_')}
    if not len(y) or not probs or any(len(p) != len(y) for p in probs.values()):
        raise ValueError('Incomplete predictions')
    reference = 'lstm_ohlc' if 'lstm_ohlc' in probs else next(iter(probs))
    out = {
        'n_test': int(len(y)), 'reference_model': reference,
        'purpose': 'post-hoc descriptive evaluation; do not select configurations or trade thresholds from this file',
        'original_trial_count': int(r['n_trials']),
        'models': {}, 'accuracy_comparisons': {},
        'confidence_thresholds': 'fixed before inspection, NOT calibrated to profitability',
        'volatility_regimes': None,
    }
    for m,p in probs.items():
        out['models'][m] = {
            'accuracy': float(((p>=.5)==y).mean()),
            'brier': float(np.mean((p-y)**2)),
            'confidence_bins': _confidence_bins(p, y),
        }
        if m != reference:
            out['accuracy_comparisons'][f'{m}_minus_{reference}'] = _comparison(p, probs[reference],y,bootstraps)
    if 'fusion' in probs and 'wm' in probs:
        out['accuracy_comparisons']['fusion_minus_wm'] = _comparison(probs['fusion'],probs['wm'],y,bootstraps)
    if data:
        df = load_candles(data)
        provenance = r.get('data_provenance', {})
        if provenance.get('sha256'):
            from src.run_experiments import _sha256
            if provenance['sha256'] != _sha256(data):
                raise ValueError('Historical CSV hash differs from experiment provenance')
        if t_idx.max() >= len(df):
            raise ValueError('Test candle indices exceed available market data')
        out['volatility_regimes'] = _historical_regimes(
            r,t_idx,df.close.to_numpy(),r['folds'],probs,y)
    else:
        out['volatility_explanation'] = 'Skipped; supply the exact historical CSV used in this experiment.'
    path = folder/'phase2_diagnostics.json'
    path.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved',path)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results_phase2')
    parser.add_argument('--data',default=None)
    parser.add_argument('--bootstrap',type=int,default=300)
    args=parser.parse_args()
    evaluate(args.results,args.data,args.bootstrap)


if __name__=='__main__':
    main()
