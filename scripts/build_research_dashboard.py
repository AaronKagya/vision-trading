"""Build an OFFLINE, historical-only Vision Trading research dashboard.

Consumes actual immutable GitHub Actions evidence and embeds a lightweight
snapshot in a single HTML file. No model calls, broker API, or network.
Usage:
 python -m scripts.build_research_dashboard --holdout-zip path/to/2025q1.zip \
   --summary-zip path/to/2024_summary.zip --out site/index.html
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

TEMPLATE = Path(__file__).resolve().parents[1] / 'site' / 'template.html'
MODEL_IDS = ('lstm_ohlc', 'logreg_lags', 'majority')


def _member(zf, suffix):
    names = [n for n in zf.namelist() if n.endswith(suffix) and not n.startswith('__MACOSX/')]
    if len(names) != 1:
        raise ValueError(f'Expected exactly one {suffix}, found {len(names)}')
    return zf.read(names[0])


def make_payload(holdout_zip: str | Path, summary_zip: str | Path):
    with zipfile.ZipFile(holdout_zip) as z:
        hold = json.loads(_member(z,'/metrics.json'))
        raw = _member(z, '/preds.npz')
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            got = set(arrays.files)
            required = {'y', 'fwd', 't_idx'} | {f'p_{m}' for m in MODEL_IDS}
            if not required.issubset(got):
                raise ValueError(f'Missing holdout prediction arrays: {required-got}')
            y = arrays['y'].astype(int)
            fwd = arrays['fwd'].astype(float)
            idx = arrays['t_idx'].astype(int)
            n = int(hold['n_test'])
            if n < 1 or any(len(a) != n for a in (y,fwd,idx)):
                raise ValueError('Holdout observations mismatch metrics')
            if (not np.isin(y,[0,1]).all() or not np.isfinite(fwd).all() or
                np.any(np.diff(idx)!=1)):
                raise ValueError('Invalid holdout labels, returns, or indices')
            preds = {}
            for model in MODEL_IDS:
                a = arrays[f'p_{model}']
                if len(a)!=n or not np.isfinite(a).all() or (a<0).any() or (a>1).any():
                    raise ValueError(f'Invalid probabilities: {model}')
                preds[model] = np.round(a,5).tolist()
    with zipfile.ZipFile(summary_zip) as z:
        summ = json.loads(_member(z,'summary.json'))
    if hold['external_test_period']!='2025-01-01 through 2025-03-31, 15-minute candles':
        raise ValueError('Unexpected 2025 holdout test period')
    if hold.get('training_must_not_use_2025') is not True:
        raise ValueError('No verified 2024-only training fence')
    if summ.get('status')!='completed' or summ.get('seeds')!=[0,1,2,3,4] or summ.get('model_count')!=8:
        raise ValueError('2024 benchmark is not complete five-seed evidence')
    if not all(m in hold['models'] for m in MODEL_IDS):
        raise ValueError('Missing holdout model scores')
    if not all(m in summ['models'] for m in ('lstm_ohlc','fusion','wm','majority')):
        raise ValueError('2024 benchmark missing expected models')
    return {
        'schema':2,
        'base_utc':'2025-01-01T00:00:00Z',
        'interval_minutes':15,
        'n_test':n,
        'y':y.tolist(),
        'forward_log_returns':np.round(fwd,8).tolist(),
        't_idx':idx.tolist(),
        'predictions':preds,
        'holdout_metrics':hold,
        'benchmark':summ,
        'sources':{
            'holdout_sha256':hashlib.sha256(Path(holdout_zip).read_bytes()).hexdigest(),
            'summary_sha256':hashlib.sha256(Path(summary_zip).read_bytes()).hexdigest(),
        }
    }


def build(holdout_zip,summary_zip,out):
    payload=make_payload(holdout_zip,summary_zip)
    template=TEMPLATE.read_text(encoding='utf-8')
    if template.count('@@EVIDENCE_JSON@@')!=1:
        raise ValueError('Template must have exactly one evidence placeholder')
    safe=json.dumps(payload,ensure_ascii=True,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    result=template.replace('@@EVIDENCE_JSON@@',safe)
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(result,encoding='utf-8')
    print(f'Created {out} ({len(result):,} chars), {payload["n_test"]} unseen historical observations')
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--holdout-zip',required=True)
    ap.add_argument('--summary-zip',required=True)
    ap.add_argument('--out',default='site/index.html')
    a=ap.parse_args()
    build(a.holdout_zip,a.summary_zip,a.out)

if __name__=='__main__': main()
