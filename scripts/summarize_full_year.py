"""Aggregate completed *predeclared* BTCUSDT 2024 five-seed benchmarks.

The five seeds reuse the *same* chronological test dates. Seed-to-seed
variation is descriptive training variability, NOT five independent samples
of financial market history. No post-hoc model selection is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

SEEDS = tuple(range(5))
MODELS = (
    'majority', 'persistence', 'logreg_lags', 'lstm_ohlc', 'cnn_static',
    'wm_nolatent', 'wm', 'fusion',
)
FIELDS = ('accuracy', 'auc', 'brier', 'sharpe_net', 'psr', 'dsr')


def collect(root: Path, seeds: tuple[int, ...] = SEEDS):
    """Validate one metrics.json per seed, identical data/configuration/folds.

    Raises ValueError rather than silently summarizing a partial benchmark.
    """
    paths = sorted(root.rglob('metrics.json'))
    if not paths:
        raise ValueError(f'No metrics.json files under {root}')
    runs: dict[int, dict] = {}
    for path in paths:
        res = json.loads(path.read_text(encoding='utf-8'))
        cfg = res['config']
        seed = cfg['seed']
        if not isinstance(seed, int) or seed not in seeds:
            raise ValueError(f'Unexpected seed {seed} at {path}')
        if seed in runs:
            raise ValueError(f'Duplicate result for seed {seed}: {path}')
        if cfg.get('models') != 'phase2' or cfg.get('folds') != 4 or cfg.get('epochs') != 10:
            raise ValueError(f'Unexpected benchmark configuration in {path}')
        for key, val in [('window',32), ('img',64), ('horizon',4), ('K',6), ('bs',128), ('bootstrap',1000), ('cost',0.0005)]:
            if cfg.get(key) != val:
                raise ValueError(f'{path}: expected {key}={val}, got {cfg.get(key)}')
        if tuple(res['models']) != MODELS:
            raise ValueError(f'{path}: model list or ordering changed')
        if len(res.get('folds', [])) != 4:
            raise ValueError(f'{path}: expected four completed walk-forward folds')
        if res.get('n_test',0) < 10000:
            raise ValueError(f'{path}: too few unseen examples for full year')
        prov=res.get('data_provenance', {})
        if prov.get('synthetic') or not prov.get('sha256'):
            raise ValueError(f'{path}: not real market data with SHA256')
        if not str(prov.get('start_utc','')).startswith('2024-01-01') or not str(prov.get('end_utc','')).startswith('2024-12-31'):
            raise ValueError(f'{path}: data period is not Jan-Dec 2024')
        if prov.get('rows') != 35136:  # leap-year 2024: 366 days x 96 candles
            raise ValueError(f'{path}: incorrect data row count')
        runs[seed] = res
    missing = sorted(set(seeds)-set(runs))
    if missing:
        raise ValueError(f'Missing completed seed results: {missing}')
    data_hashes = {r['data_provenance']['sha256'] for r in runs.values()}
    if len(data_hashes) != 1:
        raise ValueError('Different input CSV SHA-256 across seeds')
    test_counts = {r['n_test'] for r in runs.values()}
    if len(test_counts) != 1:
        raise ValueError('Unseen test period sample count differs by seed')
    signatures = [{k: v for k,v in r['config'].items() if k not in ('seed','out')}
                  for r in runs.values()]
    if any(c != signatures[0] for c in signatures[1:]):
        raise ValueError('Experiment configuration differs across seeds')
    return runs


def summarize(runs: dict[int,dict]) -> dict:
    ordered = [runs[s] for s in sorted(runs)]
    out = {
        'status': 'completed',
        'seeds': sorted(runs),
        'data_sha256': ordered[0]['data_provenance']['sha256'],
        'rows': ordered[0]['data_provenance']['rows'],
        'n_test_per_seed': ordered[0]['n_test'],
        'model_count': len(MODELS),
        'walk_forward_folds': 4,
        'warnings': [
            'All seeds share identical historical test periods; variability across seeds is NOT independent market replication.',
            'Accuracy does not establish profitability; annualized net Sharpe and DSR are research proxies.',
            'No deployment/live execution, no V75 data, no claim of trading skill.',
            'A fusion model is an additional tested configuration and must be counted in model selection adjustments.',
        ],
        'models': {},
    }
    for model in MODELS:
        details = {}
        for field in FIELDS:
            v = [r['models'][model][field] for r in ordered]
            details[field+'_mean']=statistics.mean(v)
            details[field+'_sd']=statistics.stdev(v) if len(v)>1 else 0.0
            details[field+'_per_seed']=v
        if model != 'lstm_ohlc':
            differences = [r['models'][model]['accuracy'] - r['models']['lstm_ohlc']['accuracy'] for r in ordered]
            details['paired_acc_diff_vs_lstm_mean']=statistics.mean(differences)
            details['paired_acc_diff_vs_lstm_per_seed']=differences
            details['seeds_beating_lstm']=sum(x>0 for x in differences)
        out['models'][model]=details
    return out


def markdown(summary: dict) -> str:
    lines=['# BTCUSDT 2024 — Phase 2 five-seed benchmark', '',
           f"**Status:** {summary['status']} · {len(summary['seeds'])} seeds · {summary['walk_forward_folds']} folds · {summary['n_test_per_seed']:,} unseen predictions per seed.",
           '', 'The model ranking below is *descriptive*. No model is approved for live trading.', '',
           '| Model | Direction accuracy mean ± seed SD | AUC mean | Net Sharpe mean | DSR mean |',
           '|---|---:|---:|---:|---:|']
    for model, vals in summary['models'].items():
        lines.append(f"| {model} | {100*vals['accuracy_mean']:.2f}% ± {100*vals['accuracy_sd']:.2f} pp | {vals['auc_mean']:.4f} | {vals['sharpe_net_mean']:.2f} | {vals['dsr_mean']:.4g} |")
    lines += ['', '## Caveats', '']+[f'- {w}' for w in summary['warnings']]
    lines += ['', f"Dataset SHA-256: `{summary['data_sha256']}`", '',
              'See individual seed `metrics.json` and `preds.npz` artifacts for paired inference and audit.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='collected')
    p.add_argument('--out',default='benchmark_summary')
    a=p.parse_args()
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    try:
        results=summarize(collect(root))
    except (ValueError,KeyError,TypeError) as exc:
        (out/'incomplete.txt').write_text(str(exc)+'\n',encoding='utf-8')
        raise SystemExit(f'Incomplete benchmark: {exc}') from exc
    (out/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (out/'summary.md').write_text(markdown(results),encoding='utf-8')
    print(markdown(results),flush=True)

if __name__=='__main__':
    main()
