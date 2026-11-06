"""Train-only epoch diagnostics for the numerical LSTM.

This module does not read, score or tune on 2025 data. Model selection
is based solely on the declared October-December 2024 validation split.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from src.features import panels, numeric_features
from src.models import LSTMBaseline, DEV, _set_seed

AUDIT_START = pd.Timestamp('2024-01-01T00:00:00Z')
VAL_START = pd.Timestamp('2024-10-01T00:00:00Z')
AUDIT_END = pd.Timestamp('2025-01-01T00:00:00Z')
BUDGETS = (5, 10, 20, 40)
HORIZON = 4
WINDOW = 32
EMBARGO = 10
PATIENCE = 3
LR = 0.001
BS = 128


def make_audit_data(df):
    """Strict year boundary and forward-label-safe train/validation split."""
    if df.time.iloc[0] != AUDIT_START or df.time.iloc[-1] != AUDIT_END - pd.Timedelta(minutes=15):
        raise ValueError('Training audit must use exactly 2024 UTC candles')
    if len(df) != 35136 or (df.time.diff().iloc[1:] != pd.Timedelta(minutes=15)).any():
        raise ValueError('Missing, duplicated or irregular 2024 candles')
    idx, pan = panels(df, window=WINDOW)
    num = numeric_features(pan)
    close = df.close.to_numpy(dtype=np.float64)
    valid = np.arange(len(idx)-HORIZON, dtype=np.int64)
    cutoff = VAL_START.value
    times = df.time.astype('int64').to_numpy()
    tr_candidates = valid[times[idx[valid]+HORIZON] < cutoff]
    tr = tr_candidates[:-EMBARGO]
    va = valid[(times[idx[valid]] >= cutoff) & (times[idx[valid]+HORIZON] < AUDIT_END.value)]
    if len(tr) < 1000 or len(va) < 1000:
        raise ValueError('Not enough training/validation samples')
    if not (idx[tr[-1]]+HORIZON < idx[va[0]] and times[idx[tr]+HORIZON].max() < cutoff):
        raise RuntimeError('Training/validation labels overlap')
    y = np.zeros(len(idx), dtype=np.int64)
    y[valid] = (close[idx[valid]+HORIZON] > close[idx[valid]]).astype(np.int64)
    return num, y, tr, va, idx


def _bce(model, data, labels, idx, bs=BS):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0,len(idx),bs):
            b = idx[start:start+bs]
            total += F.binary_cross_entropy_with_logits(model(data[b]),labels[b],reduction='sum').item()
    return float(total/len(idx))


def fit_audit(num, labels, tr, va, *, seed=0, budgets=BUDGETS,
              max_epochs=40, patience=PATIENCE, bs=BS, lr=LR):
    """One complete 40-epoch trajectory, snapshot validation-best state for each cap.

    Train to 40 *regardless* of early-stopping triggers, recording the
    first epoch when old patience=3 would have stopped. Otherwise an
    early stop at epoch 4 could make 5/10/20/40 comparisons identical
    and hide whether later training would improve validation loss.
    """
    budgets=tuple(sorted(set(int(b) for b in budgets)))
    if not budgets or budgets[0]<1 or max_epochs<budgets[-1] or patience<1 or bs<2:
        raise ValueError('Invalid epoch budget or train parameters')
    if not len(tr) or not len(va) or max(tr)>=min(va):
        raise ValueError('Training and validation partitions must be chronological and nonempty')
    _set_seed(seed)
    rng=np.random.default_rng(seed)
    sd=num[tr].reshape(-1,num.shape[-1]).std(axis=0)+1e-9
    xx=torch.as_tensor(num/sd,device=DEV,dtype=torch.float32)
    yy=torch.as_tensor(labels,device=DEV,dtype=torch.float32)
    model=LSTMBaseline(num.shape[-1]).to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    best_loss=float('inf')
    best_epoch=0
    best_state=None
    stale=0
    history=[]
    snapshots={}
    early_stopping_trigger_epoch=None
    for epoch in range(1,max_epochs+1):
        model.train()
        perm=rng.permutation(tr)
        for start in range(0,len(perm),bs):
            b=perm[start:start+bs]
            if len(b)<2:continue
            loss=F.binary_cross_entropy_with_logits(model(xx[b]),yy[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        train_loss=_bce(model,xx,yy,tr,bs)
        val_loss=_bce(model,xx,yy,va,bs)
        if not np.isfinite(train_loss+val_loss):raise FloatingPointError('Nonfinite train or validation loss')
        improved = val_loss < best_loss-1e-4
        if improved:
            best_loss=val_loss
            best_epoch=epoch
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
            stale=0
        else:stale+=1
        if stale>=patience and early_stopping_trigger_epoch is None:
            early_stopping_trigger_epoch=epoch
        history.append({'epoch':epoch,'train_bce':train_loss,'validation_bce':val_loss,
                        'best_epoch_to_date':best_epoch,'best_validation_bce_to_date':best_loss,
                        'improved':improved,'old_patience_stopping_triggered':epoch==early_stopping_trigger_epoch})
        if epoch in budgets:
            snapshots[epoch]={'state_dict':copy.deepcopy(best_state),'selected_epoch':best_epoch,
                              'best_validation_bce':best_loss}
        print(f'seed={seed} epoch={epoch} train_BCE={train_loss:.6f} val_BCE={val_loss:.6f} best={best_epoch}',flush=True)
    for budget in budgets:
        if budget not in snapshots:
            snapshots[budget]={'state_dict':copy.deepcopy(best_state),'selected_epoch':best_epoch,
                               'best_validation_bce':best_loss}
    return history, snapshots, sd


def save_checkpoints(folder, snapshots, sd, *, config, data_sha256):
    path=Path(folder)
    path.mkdir(parents=True,exist_ok=True)
    for budget,item in snapshots.items():
        torch.save({'state_dict':item['state_dict'],'scaler_std':torch.as_tensor(sd),
                    'selected_epoch':item['selected_epoch'],'best_validation_bce':item['best_validation_bce'],
                    'model':'LSTMBaseline','data_sha256_2024':data_sha256,'config':config},
                    path/f'checkpoint_budget_{budget}.pt')


def predict_checkpoint(path, num, test_idx, *, expected_budget, expected_data_sha):
    """Only inference. Never refit scaler or use test labels."""
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['config']['budgets'] != list(BUDGETS) or ck['data_sha256_2024'] != expected_data_sha:
        raise ValueError('Checkpoint configuration or 2024 data fingerprint mismatch')
    if ck['config']['seed'] < 0 or ck['config']['max_epochs'] != max(BUDGETS):
        raise ValueError('Unexpected checkpoint training configuration')
    if Path(path).name != f'checkpoint_budget_{expected_budget}.pt':
        raise ValueError('Not the selected validation-only epoch budget')
    scaler=ck['scaler_std'].detach().numpy()
    if not np.isfinite(scaler).all() or (scaler<=0).any():raise ValueError('Invalid checkpoint scaler')
    model=LSTMBaseline(num.shape[-1]).to(DEV)
    model.load_state_dict(ck['state_dict']);model.eval()
    out=[]
    with torch.no_grad():
        for st in range(0,len(test_idx),512):
            t=test_idx[st:st+512]
            inputs=torch.as_tensor(num[t]/scaler,device=DEV,dtype=torch.float32)
            out.append(torch.sigmoid(model(inputs)).cpu().numpy())
    return np.concatenate(out)
