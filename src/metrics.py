"""Forecast scores and cautious research-only net trading proxies.

No brokerage fills, financing, funding, slippage, margin or liquidation model.
Net log P&L assumes a linear +/- log-return exposure. It is a research
approximation, NOT an executable equity curve, especially for shorts.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
EULER = 0.5772156649

def ece(p,y,bins=10):
    p=np.asarray(p,float);y=np.asarray(y,float)
    edges=np.linspace(0,1,bins+1)
    score=0.
    for i in range(bins):
        idx=(p>=edges[i])&((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if idx.any():score+=idx.mean()*abs(p[idx].mean()-y[idx].mean())
    return float(score)

def forecast_metrics(p,y):
    p=np.asarray(p,float); y=np.asarray(y,dtype=int)
    if p.shape!=y.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Probability array must match labels and contain valid probabilities')
    a=float(((p>=.5)==y).mean());b=float(np.mean((p-y)**2))
    if len(np.unique(y))!=2:
        raise ValueError('ROC AUC is undefined with only one holdout class; use a longer test period')
    auc=float(roc_auc_score(y,p))
    return {'accuracy':a,'brier':b,'auc':auc,'ece':ece(p,y)}

def strategy_returns(p,fwd,cost=.0005,stride=1):
    """Original-research-compatible log P&L proxy. Starts flat, omits end liquidation.

    Every stride-th sample is a NON-overlapping horizon-h prediction. Transaction
    costs are per unit position change, including 2 units on a long/short flip.
    """
    p=np.asarray(p,float);fwd=np.asarray(fwd,float)
    if stride<1 or len(p)!=len(fwd) or not np.isfinite(fwd).all() or cost<0:
        raise ValueError('invalid trade series or costs')
    pos=np.where(p[::stride]>=.5,1.,-1.)
    turn=np.abs(np.diff(np.r_[0.,pos]))
    return pos*fwd[::stride]-cost*turn

def sharpe(r, periods_per_year):
    r=np.asarray(r,float)
    sd=r.std(ddof=1) if len(r)>1 else 0.
    return float(r.mean()/sd*np.sqrt(periods_per_year)) if sd>0 else 0.

def psr(r,sr_star=0.):
    r=np.asarray(r,float)
    if len(r)<4:return float('nan')
    std=r.std(ddof=1)
    if std<=1e-15:return 0. if r.mean()<=0 else 1.
    sr=r.mean()/std
    skew=stats.skew(r,bias=False); kurt=stats.kurtosis(r,fisher=False,bias=False)
    denominator=np.sqrt(max(1-skew*sr+(kurt-1)*sr**2/4,1e-9))
    return float(stats.norm.cdf((sr-sr_star)*np.sqrt(len(r)-1)/denominator))

def deflated_sharpe(r,trial_srs,n_trials):
    """Approximate DSR given *recorded* trial returns; not proof all selection is corrected.

    When n_trials > len(trial_srs), assumes observed trial-SR variance applies
    to unrecorded trials. All configurations tried must be separately logged.
    """
    t=np.asarray(trial_srs,float)
    n=max(2,int(n_trials))
    var=float(np.var(t,ddof=1)) if len(t)>1 else 0.
    expected_max=np.sqrt(var)*((1-EULER)*stats.norm.ppf(1-1/n)+EULER*stats.norm.ppf(1-1/(n*np.e)))
    return psr(r,expected_max)

def block_bootstrap(stat_fn,n,block=50,B=1000,seed=0):
    if n<1 or B<2: raise ValueError('n must be positive and B>=2')
    block=max(1,min(int(block),n));rng=np.random.default_rng(seed)
    blocks=int(np.ceil(n/block));out=np.empty(B,float)
    for i in range(B):
        starts=rng.integers(0,n-block+1,size=blocks)
        idx=np.concatenate([np.arange(a,a+block) for a in starts])[:n]
        out[i]=stat_fn(idx)
    return [float(x) for x in np.percentile(out,[2.5,97.5])]

def compounded_from_log_proxy(log_sum):
    return float(np.expm1(log_sum))
