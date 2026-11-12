"""Aggregate verified cross-instrument 2024 experiments without inventing data.

Initial preview may contain only verified BTCUSDT metrics. Normal runs require
ALL ETHUSDT/SOLUSDT seed artifacts and reject missing/mismatched metadata.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from datetime import datetime, timezone

EXPECTED_MODELS = ('majority', 'persistence', 'logreg_lags', 'lstm_ohlc',
                   'cnn_static', 'wm_nolatent', 'wm', 'fusion')
NEW_SYMBOLS = ('ETHUSDT', 'SOLUSDT')
PARAMETERS = {'models': 'phase2', 'folds': 4, 'epochs': 10,
              'window': 32, 'img': 64, 'K': 6, 'horizon': 4,
              'bs': 128, 'bootstrap': 1000, 'cost': .0005}
METRICS = ('accuracy', 'auc', 'brier', 'sharpe_net')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_btc(btc):
    validate(btc, 0, 'BTCUSDT', check_hash=False)
    return btc


def validate(result, seed, symbol, *, check_hash=True):
    cfg = result['config']
    if cfg.get('seed') != seed:
        raise ValueError(f'{symbol}: incorrect seed {cfg.get("seed")!r}')
    for key, val in PARAMETERS.items():
        if cfg.get(key) != val:
            raise ValueError(f'{symbol}: expected {key}={val}, got {cfg.get(key)}')
    if list(result['models']) != list(EXPECTED_MODELS):
        raise ValueError(f'{symbol}: incomplete/changed model list')
    if len(result.get('folds', [])) != 4 or result.get('n_test') != 17308:
        raise ValueError(f'{symbol}: walk-forward fold count or test count differs')
    prov = result['data_provenance']
    sha = prov.get('sha256', '')
    if prov.get('synthetic') or not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(f'{symbol}: missing real-data fingerprint')
    if prov.get('rows') != 35136 or not prov.get('start_utc', '').startswith('2024-01-01') or not prov.get('end_utc', '').startswith('2024-12-31'):
        raise ValueError(f'{symbol}: invalid 2024 coverage')
    if check_hash and not cfg.get('data', '').endswith(f'{symbol}_15m.csv'):
        raise ValueError(f'{symbol}: wrong source file path')
    for m in EXPECTED_MODELS:
        for field in METRICS:
            value = result['models'][m][field]
            if not isinstance(value, (int,float)) or not math.isfinite(value):
                raise ValueError(f'{symbol}: invalid {m}.{field}')
        if not 0 <= result['models'][m]['accuracy'] <= 1:
            raise ValueError(f'{symbol}: invalid accuracy')
    return result


def as_market(results, symbol):
    signatures = {r['data_provenance']['sha256'] for r in results.values()}
    if len(signatures) != 1:
        raise ValueError(f'{symbol}: inconsistent dataset hashes')
    models = {}
    for model in EXPECTED_MODELS:
        entry = {}
        for field in METRICS:
            arr = [v['models'][model][field] for _,v in sorted(results.items())]
            entry[field + '_mean'] = mean(arr)
            entry[field + '_sd'] = stdev(arr) if len(arr)>1 else None
        models[model] = entry
    return {'status':'measured', 'seeds':sorted(results), 'folds':4,
            'n_test_per_seed':17308, 'rows':35136, 'data_sha256':next(iter(signatures)),
            'models':models}


def from_btc_summary(btc_summary):
    if (btc_summary.get('status') != 'completed' or btc_summary.get('seeds') != [0,1,2,3,4]
        or btc_summary.get('model_count') != 8 or btc_summary.get('walk_forward_folds') != 4
        or btc_summary.get('n_test_per_seed') != 17308 or btc_summary.get('rows') != 35136):
        raise ValueError('Invalid original BTCUSDT five-seed summary')
    if set(btc_summary['models']) != set(EXPECTED_MODELS):
        raise ValueError('Wrong BTCUSDT summary models')
    models={}
    for name in EXPECTED_MODELS:
        models[name]={}
        for field in METRICS:
            models[name][field+'_mean']=btc_summary['models'][name][field+'_mean']
            models[name][field+'_sd']=btc_summary['models'][name][field+'_sd']
    return {'status':'measured','seeds':[0,1,2,3,4],'folds':4,
            'n_test_per_seed':17308,'rows':35136,'data_sha256':btc_summary['data_sha256'],'models':models}


def aggregate(root, btc0, btcsummary, seeds=(0,), *, initial=False):
    seeds=tuple(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or not set(seeds).issubset(set(range(5))):
        raise ValueError('Invalid seed selection')
    if seeds!=(0,) and seeds!=(0,1,2,3,4):
        raise ValueError('Supported comparisons: pilot seed 0 or full seeds 0..4')
    if initial and seeds != (0,):
        raise ValueError('Initial preview supports pilot only')
    base=check_btc(read(btc0))
    all_markets = {
        'BTCUSDT':as_market({0:base},'BTCUSDT') if len(seeds)==1 else from_btc_summary(read(btcsummary))
    }
    found={symbol:{} for symbol in NEW_SYMBOLS}
    if not initial:
        paths=list(Path(root).rglob('metrics.json'))
        if len(paths) != len(NEW_SYMBOLS)*len(seeds):
            raise ValueError(f'Expected {len(NEW_SYMBOLS)*len(seeds)} result files, got {len(paths)}')
        for path in paths:
            result=read(path)
            source=str(result.get('config',{}).get('data',''))
            symbols=[s for s in NEW_SYMBOLS if source.endswith(f'{s}_15m.csv')]
            if len(symbols)!=1:
                raise ValueError(f'Unexpected market source in {path}')
            sym=symbols[0]; seed=result['config']['seed']
            if seed not in seeds or seed in found[sym]:
                raise ValueError(f'Duplicate or unexpected {sym} seed {seed}')
            found[sym][seed]=validate(result,seed,sym)
    for sym in NEW_SYMBOLS:
        if initial:
            all_markets[sym]={'status':'pending', 'note':'Real-data benchmark not completed; no measured performance yet.'}
        else:
            if set(found[sym])!=set(seeds):
                raise ValueError(f'{sym}: incomplete seeds {set(seeds)-set(found[sym])}')
            all_markets[sym]=as_market(found[sym],sym)
    for sym in ('XAUUSD','EURUSD','VOLATILITY75'):
        all_markets[sym]={'status':'planned','note':'No comparable validated dataset or experiment yet.'}
    return {'schema_version':1,'status':'baseline_only' if initial else ('pilot' if len(seeds)==1 else 'five_seed'),
            'generated_utc':datetime.now(timezone.utc).isoformat(),
            'study_period':'2024-01-01 through 2024-12-31, UTC',
            'timeframe':'15m','prediction_horizon_candles':4,'models':list(EXPECTED_MODELS),
            'parameters':PARAMETERS,'markets':all_markets,
            'warnings':[
                'Models use historical BTCUSDT, ETHUSDT and SOLUSDT Binance SPOT candles, not live trading.',
                'Markets share wall-clock periods; cross-asset results are correlated, not independent replications.',
                'Across-market percentage accuracy is not an executable profit comparison.',
                'The same illustrative turnover cost of 0.0005 is not a validated, asset-specific execution model.',
                'Different random seeds reuse identical market test windows; seed variability is not an independent test set.',
                'No model has been authorized for live trading and no profitability is claimed.'
            ]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='collected_multiasset')
    p.add_argument('--btc0',default='evidence/btcusdt_2024_seed0_metrics.json')
    p.add_argument('--btcsummary',default='evidence/btcusdt_2024_five_seed_summary.json')
    p.add_argument('--seeds',nargs='+',type=int,default=[0])
    p.add_argument('--initial',action='store_true')
    p.add_argument('--out',default='site/compare/data.json')
    args=p.parse_args()
    result=aggregate(args.root,args.btc0,args.btcsummary,tuple(args.seeds),initial=args.initial)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'Saved {result["status"]} comparison to {path}',flush=True)


if __name__ == '__main__':
    main()
