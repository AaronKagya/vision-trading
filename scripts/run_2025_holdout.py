"""Locked external holdout: 2024 trains/validates a numerical LSTM, 2025-Q1 tests.

Never uses 2025 for fitting, model selection, normalization, threshold choice,
or hyperparameter tuning. BTCUSDT SPOT only, 15-minute continuous UTC candles.
This is a limited research test, not live trading or a brokerage backtest.
"""
from __future__ import annotations
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.data import load_candles
from src.features import panels,numeric_features
from src.models import fit_predict_lstm
from src.metrics import forecast_metrics
from src.abstention import evaluate_policy,DEFAULT_DISTANCE

TRAIN_START = pd.Timestamp('2024-01-01T00:00:00Z')
TEST_START = pd.Timestamp('2025-01-01T00:00:00Z')
TEST_END = pd.Timestamp('2025-04-01T00:00:00Z')


def ensure_locked_range(df):
    if df.time.iloc[0] != TRAIN_START or df.time.iloc[-1] != TEST_END - pd.Timedelta(minutes=15):
        raise ValueError('Holdout needs exactly 2024-01-01 through 2025-03-31 23:45 UTC')
    expected=int((TEST_END-TRAIN_START)/pd.Timedelta(minutes=15))
    if len(df)!=expected:raise ValueError(f'Incomplete locked period: expected {expected} candles, got {len(df)}')


def split_locked(df,t_idx,h=4,embargo=10):
    """2025 never contributes to fitting. 2024 labels end before Jan 1, 2025."""
    if h<1 or embargo<h:raise ValueError('Invalid horizon or embargo')
    t_idx=np.asarray(t_idx,dtype=int)
    # Pandas may emit timezone-aware datetime arrays. Use integer ns instead.
    ns=df.time.astype('int64').to_numpy()
    boundary=TEST_START.value;finish=TEST_END.value
    eligible=np.arange(len(t_idx)-h)
    candidate_train=eligible[ns[t_idx[eligible]+h] < boundary]
    candidate_test=eligible[(ns[t_idx[eligible]]>=boundary) & (ns[t_idx[eligible]+h]<finish)]
    if len(candidate_train)<1000 or len(candidate_test)<1000:
        raise ValueError('Insufficient train/test rows')
    val_start=int(len(candidate_train)*.85)
    train=candidate_train[:val_start-embargo]
    valid=candidate_train[val_start:]
    test=candidate_test
    if not len(train) or not len(valid) or not len(test):raise ValueError('Empty split')
    if not (t_idx[train[-1]]+h < t_idx[valid[0]] and t_idx[valid[-1]]+h < t_idx[test[0]]):
        raise ValueError('Train/validation/test label leakage')
    assert ns[t_idx[train]+h].max() < boundary
    assert ns[t_idx[valid]+h].max() < boundary
    assert ns[t_idx[test]].min()>=boundary
    return train,valid,test


def run(source,out='results_holdout_2025q1',seed=0,epochs=10,batch_size=128):
    import torch
    if epochs<1 or batch_size<2:raise ValueError('Invalid training parameters')
    df=load_candles(source)
    ensure_locked_range(df)
    t_idx,pan=panels(df,window=32)
    num=numeric_features(pan)
    h=4
    close=df.close.to_numpy()
    forward=np.log(close[t_idx[h:]]/close[t_idx[:-h]])
    y=(forward>0).astype(int)
    tr,va,te=split_locked(df,t_idx,h=h,embargo=10)
    # Num and target must share full panel length; trailing labels do not exist.
    yall=np.zeros(len(t_idx),dtype=int);yall[:len(y)]=y
    pred_lstm=fit_predict_lstm(num,yall,tr,va,te,epochs,0.001,batch_size,seed)
    closes=close
    lag=np.stack([np.log(closes[t_idx-k]/closes[t_idx-k-1]) for k in range(5)],axis=1)
    logreg=LogisticRegression(max_iter=500,random_state=seed)
    logreg.fit(lag[tr],yall[tr])
    scores={'lstm_ohlc':pred_lstm,'logreg_lags':logreg.predict_proba(lag[te])[:,1],
            'majority':np.full(len(te),float(yall[tr].mean()>=.5))}
    r_fwd=forward[te]
    result={
        'dataset_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'data_first_utc':df.time.iloc[0].isoformat(),'data_last_utc':df.time.iloc[-1].isoformat(),
        'training_period':'2024-01-01 through 2024-12-31, 15-minute candles',
        'external_test_period':'2025-01-01 through 2025-03-31, 15-minute candles',
        'seed':seed,'epochs':epochs,'horizon':h,'window':32,'embargo':10,
        'n_train':len(tr),'n_validation':len(va),'n_test':len(te),
        'confidence_distance':DEFAULT_DISTANCE,'transaction_cost_per_unit_change':.0005,
        'models':{},'limitations':'No live trading; P&L is a log-price directional proxy, no fills/funding/short financing.',
        'training_must_not_use_2025':True,'python_version':platform.python_version(),'torch_version':torch.__version__}
    for name,p in scores.items():
        r=forecast_metrics(p,yall[te]);
        r['abstain_policy']=evaluate_policy(p,yall[te],r_fwd,horizon=h,distance=DEFAULT_DISTANCE,cost=.0005)
        r['always_position_proxy']=evaluate_policy(p,yall[te],r_fwd,horizon=h,distance=0.,cost=.0005)
        result['models'][name]=r
    target=Path(out);target.mkdir(parents=True,exist_ok=True)
    (target/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    np.savez_compressed(target/'preds.npz',y=yall[te],fwd=r_fwd,t_idx=t_idx[te],**{'p_'+k:v for k,v in scores.items()})
    markdown=['# BTCUSDT 2025-Q1 locked holdout','',
      'Training and model selection use 2024 only. January–March 2025 is the external test period.','',
      '| Model | Accuracy | AUC | Abstain coverage | Conditional accuracy | Net log-P&L proxy |',
      '|---|---:|---:|---:|---:|---:|']
    for name,m in result['models'].items():
        ab=m['abstain_policy']; sel=ab['selected_direction_accuracy']
        markdown.append(f'| {name} | {m["accuracy"]:.2%} | {m["auc"]:.3f} | {ab["coverage"]:.2%} | {sel:.2%} | {ab["net_log_pnl_proxy"]:+.4f} |' if sel is not None else f'| {name} | {m["accuracy"]:.2%} | {m["auc"]:.3f} | 0% | N/A | 0 |')
    markdown += ['','Directional proxy only, not executable backtesting or investment advice.','Threshold .05 was declared before looking at 2025.']
    (target/'report.md').write_text('\n'.join(markdown)+'\n')
    print((target/'report.md').read_text())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/BTCUSDT_15m.csv')
    p.add_argument('--out',default='results_holdout_2025q1')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--bs',type=int,default=128)
    a=p.parse_args()
    run(a.data,a.out,a.seed,a.epochs,a.bs)

if __name__=='__main__':main()
