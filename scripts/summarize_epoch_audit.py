"""Lock maximum epoch budget using ONLY five 2024 training/validation audits."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
BUDGETS = (5, 10, 20, 40)  # Locked protocol; do not change post-holdout

SEEDS=tuple(range(5))


def collect(root):
    root=Path(root)
    items=[]
    for seed in SEEDS:
        d=root/f'epoch-seed-{seed}'
        p=d/'audit.json'
        if not p.is_file():raise ValueError(f'Missing seed {seed} audit')
        item=json.loads(p.read_text())
        if item.get('seed')!=seed or item.get('config',{}).get('seed')!=seed:
            raise ValueError('Seed identity mismatch')
        for budget in BUDGETS:
            if not (d/f'checkpoint_budget_{budget}.pt').is_file():
                raise ValueError(f'Missing checkpoint for seed {seed}, cap {budget}')
        if sorted(map(int,item.get('budgets',{})))!=list(BUDGETS):raise ValueError('Incomplete budgets')
        if not item.get('history') or item.get('epochs_executed')!=len(item['history']):
            raise ValueError('Missing epoch loss history')
        history=item['history']
        if [x['epoch'] for x in history]!=list(range(1,len(history)+1)):
            raise ValueError('Nonconsecutive epochs')
        for budget in BUDGETS:
            x=item['budgets'][str(budget)]
            if not (1<=x['selected_epoch']<=min(budget,len(history))):raise ValueError('Invalid best epoch')
            if not np.isfinite(x['validation_bce']):raise ValueError('Nonfinite loss')
            if not np.isclose(x['validation_bce'],history[x['selected_epoch']-1]['validation_bce'],atol=0.00015):
                raise ValueError('Invalid checkpoint validation loss')
        items.append(item)
    base=items[0]
    # Ignore only seed-specific fields, never design parameters.
    for item in items[1:]:
        if item['data_sha256_2024'] != base['data_sha256_2024'] or item['data_rows']!=base['data_rows']:
            raise ValueError('Mixed data provenance')
        c={k:v for k,v in item['config'].items() if k!='seed'}
        b={k:v for k,v in base['config'].items() if k!='seed'}
        if c!=b:raise ValueError('Mixed model configuration')
        if item['n_train']!=base['n_train'] or item['n_validation']!=base['n_validation']:
            raise ValueError('Different split cardinality')
    values={str(k):{'mean_validation_bce':float(np.mean([a['budgets'][str(k)]['validation_bce'] for a in items])),
                    'sd_validation_bce':float(np.std([a['budgets'][str(k)]['validation_bce'] for a in items],ddof=1)),
                    'selected_epoch_per_seed':[a['budgets'][str(k)]['selected_epoch'] for a in items]} for k in BUDGETS}
    # Primary selection criterion declared before holdout: minimize mean validation BCE.
    selected=min(BUDGETS,key=lambda k:(values[str(k)]['mean_validation_bce'],k))
    return {'seeds':list(SEEDS),'selected_max_epochs':selected,'candidate_budgets':values,
            'data_sha256_2024':base['data_sha256_2024'],'data_rows_2024':base['data_rows'],
            'config_without_seed':{k:v for k,v in base['config'].items() if k!='seed'},
            'criterion':'minimum five-seed mean 2024 validation BCE, ties shorter budget',
            'test_period_not_used':'2025-04-01 through 2025-06-30'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='collected_epoch')
    p.add_argument('--out',default='selected_epoch_budget')
    a=p.parse_args()
    out=collect(a.root);target=Path(a.out);target.mkdir(parents=True,exist_ok=True)
    (target/'selected.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    msg=['# Training audit — 2024 validation only','','| Max epochs | Mean validation BCE | SD |','|---:|---:|---:|']
    for k,v in out['candidate_budgets'].items():msg.append(f'| {k} | {v["mean_validation_bce"]:.6f} | {v["sd_validation_bce"]:.6f} |')
    msg+=['',f'**Selected cap: {out["selected_max_epochs"]} epochs**, on 2024 validation only.','',
          'The checkpoint for each seed retains its validation-best epoch, which can be fewer than the maximum budget.','',
          'This report does not contain out-of-sample 2025 results or trading claims.']
    (target/'summary.md').write_text('\n'.join(msg)+'\n')
    print((target/'summary.md').read_text(),flush=True)

if __name__=='__main__':main()
