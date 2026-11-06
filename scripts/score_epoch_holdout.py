"""Apply 2024-validation-selected LSTM checkpoints to untouched 2025 Q2.

2025-Q1 is used only as trailing feature context on 2025-04-01. There
is no 2025 data in training, model selection, or normalization.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_candles
from src.features import panels,numeric_features
from src.epoch_audit import predict_checkpoint, WINDOW, HORIZON, AUDIT_START, VAL_START, AUDIT_END
from src.abstention import evaluate_policy,DEFAULT_DISTANCE
from src.metrics import forecast_metrics,block_bootstrap
from scripts.prepare_btcusdt import validate_period
from scripts.summarize_epoch_audit import collect,SEEDS

TEST_START=pd.Timestamp('2025-04-01T00:00:00Z')
TEST_END=pd.Timestamp('2025-07-01T00:00:00Z')


def first_2024_sha(full_source, n=35136):
    """Raw CSV's complete 2024 byte prefix, before 2025 data was appended."""
    digest=hashlib.sha256()
    with Path(full_source).open('rb') as f:
        for i in range(n+1):
            line=f.readline()
            if not line:raise ValueError('Insufficient raw rows for the 2024 fingerprint')
            digest.update(line)
    return digest.hexdigest()


def score(source, root, selection, out):
    root=Path(root); selection=Path(selection)
    locked=collect(root) # Revalidate all five seed histories and checkpoints.
    published=json.loads(selection.read_text())
    if published != locked:raise ValueError('Selected configuration does not match 2024-only audit')
    budget=locked['selected_max_epochs']
    df=load_candles(source)
    validate_period(df,'2024-01','2025-06')
    sha_2024=first_2024_sha(source)
    if sha_2024 != locked['data_sha256_2024']:
        raise ValueError('2024 training-data prefix differs from archived model data')
    idx,pan=panels(df,window=WINDOW)
    num=numeric_features(pan)
    times=df.time.astype('int64').to_numpy()
    eligible=np.arange(len(idx)-HORIZON)
    te=eligible[(times[idx[eligible]] >= TEST_START.value) &
                (times[idx[eligible]+HORIZON] < TEST_END.value)]
    if len(te)<7000 or df.time.iloc[idx[te[0]]] != TEST_START or (df.time.iloc[idx[te[-1]]]+pd.Timedelta(hours=1))>=TEST_END:
        raise ValueError('Incomplete or misaligned Q2 test')
    if not (times[idx[te]].min() >= TEST_START.value and times[idx[te]+HORIZON].max() < TEST_END.value):
        raise RuntimeError('Out-of-period test leakage')
    close=df.close.to_numpy(dtype=np.float64)
    fwd=np.log(close[idx[te]+HORIZON]/close[idx[te]])
    y=(fwd>0).astype(int)
    # Majority direction is fixed from 2024 training only, before October validation.
    tr=eligible[times[idx[eligible]+HORIZON] < VAL_START.value]
    tr=tr[:-10]
    majority=float(np.mean(close[idx[tr]+HORIZON] > close[idx[tr]]) >= .5)
    base=np.full(len(te),majority)
    base_scores=forecast_metrics(base,y)
    seeds={}
    all_acc=[]
    for seed in SEEDS:
        ck=root/f'epoch-seed-{seed}'/f'checkpoint_budget_{budget}.pt'
        p=predict_checkpoint(ck,num,te,expected_budget=budget,expected_data_sha=sha_2024)
        metrics=forecast_metrics(p,y)
        policy=evaluate_policy(p,y,fwd,horizon=HORIZON,distance=DEFAULT_DISTANCE,cost=.0005)
        # 4x15m horizon overlap removed for the auxiliary accuracy check.
        nonoverlap_accuracy=float(((p[::HORIZON]>=.5)==y[::HORIZON]).mean())
        scores={**metrics,'nonoverlapping_accuracy':nonoverlap_accuracy,
                'abstain_policy':policy,'seed':seed,'best_epoch':locked['candidate_budgets'][str(budget)]['selected_epoch_per_seed'][seed]}
        seeds[str(seed)]=scores;all_acc.append(metrics['accuracy'])
    delta=lambda sample: float(((np.asarray(p)[sample]>=.5)==y[sample]).mean()-((base[sample]>=.5)==y[sample]).mean())
    # Use separate seed 0 as an illustrative paired day-block accuracy interval only.
    p0=predict_checkpoint(root/'epoch-seed-0'/f'checkpoint_budget_{budget}.pt',num,te,
                          expected_budget=budget,expected_data_sha=sha_2024)
    day_ci=block_bootstrap(lambda sample:float(((p0[sample]>=.5)==y[sample]).mean()-((base[sample]>=.5)==y[sample]).mean()),
                           len(te),block=96,B=500,seed=617)
    result={'status':'external_test_completed','asset':'Binance BTCUSDT spot','timeframe':'15m',
            'training_validation_period':'2024 only','validation_budget_selection':budget,
            'holdout_period':'2025-04-01 to 2025-06-30, UTC',
            'n_overlap_forecasts':int(len(te)),'n_nonoverlapping':int(len(y[::HORIZON])),
            'data_sha256_full':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'data_sha256_2024_matched':sha_2024,
            'majority_training_only':base_scores,'majority_nonoverlapping_accuracy':float(((base[::HORIZON]>=.5)==y[::HORIZON]).mean()),
            'mean_lstm_accuracy':float(np.mean(all_acc)),'sd_lstm_accuracy':float(np.std(all_acc,ddof=1)),
            'seed0_paired_day_block_accuracy_delta_ci':day_ci,'models':seeds,
            'notes':['Max epoch cap selected before inspecting 2025-Q2 labels.',
                     'All five model seeds are 2024-trained and use a train-only scaling checkpoint.',
                     '2025-Q1 candles are allowed only as causal past context for April 2025 features.',
                     'This is a log-return exposure proxy, not executable trade P&L or evidence of profitability.']}
    target=Path(out);target.mkdir(parents=True,exist_ok=True)
    (target/'holdout.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['# Locked LSTM epoch audit: 2025 Q2 external evaluation','',
           f'Validation-chosen maximum epoch budget: **{budget}** (2024-only).','',
           '| Seed | Best epoch | Direction accuracy | AUC | Brier | Abstention coverage | Net log-P&L proxy |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for seed,s in seeds.items():
        ab=s['abstain_policy']
        lines.append(f'| {seed} | {s["best_epoch"]} | {s["accuracy"]:.2%} | {s["auc"]:.4f} | {s["brier"]:.4f} | {ab["coverage"]:.2%} | {ab["net_log_pnl_proxy"]:+.4f} |')
    lines += ['',f'Mean LSTM direction accuracy: {result["mean_lstm_accuracy"]:.2%}.',
              f'Majority training-only baseline: {base_scores["accuracy"]:.2%}.','',
              'These metrics must not be used for further hyperparameter selection while calling 2025 Q2 untouched.',
              'No live execution or broker-specific costs are modeled.']
    (target/'holdout.md').write_text('\n'.join(lines)+'\n')
    print((target/'holdout.md').read_text(),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/BTCUSDT_15m.csv')
    parser.add_argument('--root',default='collected_epoch')
    parser.add_argument('--selection',default='selected_epoch_budget/selected.json')
    parser.add_argument('--out',default='results_epoch_holdout_2025q2')
    x=parser.parse_args()
    score(x.data,x.root,x.selection,x.out)

if __name__=='__main__':main()
