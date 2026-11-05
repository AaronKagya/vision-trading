"""Five-seed replication of a FIXED model configuration, without test-set tuning.

  python run_seeds.py --data data/BTCUSDT_15m.csv --epochs 10 --seeds 0 1 2 3 4
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np


def _provided_options(args):
    """Read the user's explicit options so stale cached experiments cannot be reused."""
    options={}
    i=0
    while i<len(args):
        token=args[i]
        if not token.startswith('--'):
            raise ValueError('Unexpected positional argument: '+token)
        key=token[2:].replace('-','_')
        if i+1>=len(args) or args[i+1].startswith('--'):
            options[key]=True;i+=1
        else:
            options[key]=args[i+1];i+=2
    return options


def _verify_reuse(previous, expected, seed):
    config=previous['config']
    if config['seed'] != seed:
        raise ValueError('Cached seed does not match requested seed')
    for key, value in expected.items():
        if key not in config:
            raise ValueError(f'Cached result is missing requested parameter: {key}')
        stored=config[key]
        try:
            candidate=(value if isinstance(stored,str) else
                       bool(value) if isinstance(stored,bool) else
                       type(stored)(value) if stored is not None else value)
        except (ValueError,TypeError):
            raise ValueError(f'Cannot compare supplied option {key}={value!r}')
        if stored!=candidate:
            raise ValueError(f'Cached result mismatch for {key}: stored {stored!r}, requested {candidate!r}')
    source=config['data']
    if not source.startswith('synthetic') and Path(source).exists():
        digest=hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if digest!=previous.get('data_provenance',{}).get('sha256'):
            raise ValueError('Cached results refer to a different CSV hash')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--out-prefix',default='results_seed')
    p.add_argument('--summary',default='results_seeds/summary.json')
    p.add_argument('--reuse-existing',action='store_true',help='Reuse files only after checking arguments and data')
    a,rest=p.parse_known_args()
    if len(a.seeds)!=len(set(a.seeds)):raise ValueError('Duplicate seed IDs')
    if '--seed' in rest or '--out' in rest:raise ValueError('Seed and output are managed by run_seeds.py')
    expected=_provided_options(rest)
    folders=[]
    for seed in a.seeds:
        folder=Path(f'{a.out_prefix}{seed}');folders.append(folder)
        target=folder/'metrics.json'
        if target.exists() and a.reuse_existing:
            previous=json.loads(target.read_text())
            _verify_reuse(previous, expected, seed)
            print('Reused configuration-checked result:',target)
            continue
        subprocess.run([sys.executable,'-m','src.run_experiments',*rest,'--seed',str(seed),'--out',str(folder)],check=True)
    runs=[json.loads((f/'metrics.json').read_text()) for f in folders]
    confs=[{k:v for k,v in run['config'].items() if k not in ('out','seed')} for run in runs]
    if any(c!=confs[0] for c in confs[1:]):raise ValueError('Different model configurations across seed runs')
    hashes=[run['data_provenance']['sha256'] for run in runs]
    if any(h!=hashes[0] for h in hashes[1:]):
        # Synthetic seeds intentionally produce different histories.
        if not runs[0]['data_provenance']['synthetic']:raise ValueError('Different historical datasets across seed runs')
    models=list(runs[0]['models'])
    out={'seeds':a.seeds,'data_hash':hashes[0],'models':{},'warning':'Historical research only; no live profit claims.'}
    for name in models:
        acc=np.array([r['models'][name]['accuracy'] for r in runs])*100
        auc=np.array([r['models'][name]['auc'] for r in runs])
        sharpe=np.array([r['models'][name]['sharpe_net'] for r in runs])
        summary={'accuracy_mean_percent':float(acc.mean()),'accuracy_sd_percent':float(acc.std(ddof=1)) if len(acc)>1 else 0.,
            'auc_mean':float(auc.mean()),'net_sharpe_mean':float(sharpe.mean()),'accuracy_per_seed_percent':acc.tolist()}
        if name in ['cnn_static','wm_nolatent','wm','fusion'] and 'lstm_ohlc' in models:
            summary['seeds_beating_lstm']=sum(r['models'][name]['accuracy']>r['models']['lstm_ohlc']['accuracy'] for r in runs)
        out['models'][name]=summary
        print(f'{name:14s} accuracy {acc.mean():.2f}% ± {acc.std(ddof=1) if len(acc)>1 else 0:.2f} pp')
    target=Path(a.summary);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2))
    print('Saved',target)
if __name__=='__main__':main()
