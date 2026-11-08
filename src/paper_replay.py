"""Deterministic historical position replay. NO BROKER ORDERS OR LIVE DATA.

One prediction every horizon (4x 15m) avoids overlapping forward labels.
Costs are illustrative per-unit turnover including terminal flattening.
The resulting cumulative *log-return proxy* is not a portfolio equity curve.
"""
from __future__ import annotations
import numpy as np
from src.abstention import positions_from_probabilities, evaluate_policy


def replay(probabilities, forward_log_returns, *, horizon=4, distance=.05, cost=.0005):
    p=np.asarray(probabilities,dtype=float)
    r=np.asarray(forward_log_returns,dtype=float)
    if p.ndim!=1 or r.shape!=p.shape or p.size==0:
        raise ValueError('Predictions and forward returns must be nonempty 1D aligned arrays')
    if not np.isfinite(r).all():
        raise ValueError('Nonfinite forward return')
    if not isinstance(horizon,int) or horizon<1:
        raise ValueError('Invalid evaluation horizon')
    if not np.isfinite(cost) or cost<0:
        raise ValueError('Invalid cost')
    positions=positions_from_probabilities(p,distance=distance)[::horizon].astype(int)
    fwd=r[::horizon]
    sampled_p=p[::horizon]
    transitions=np.abs(np.diff(np.r_[0,positions])).astype(float)
    if len(transitions):transitions[-1]+=abs(positions[-1])
    gross=positions*fwd
    cost_paid=cost*transitions
    net=gross-cost_paid
    summary={
        'n_intervals':int(len(positions)),
        'n_active':int(np.count_nonzero(positions)),
        'turnover':float(transitions.sum()),
        'gross_log_pnl_proxy':float(gross.sum()),
        'estimated_cost_drag':float(cost_paid.sum()),
        'net_log_pnl_proxy':float(net.sum()),
        'cost_per_unit_change':float(cost),
        'confidence_threshold':float(distance),
        'n_longs':int(np.sum(positions==1)),
        'n_shorts':int(np.sum(positions==-1)),
        'n_flats':int(np.sum(positions==0)),
        'max_drawdown_log_proxy':float(np.min(np.cumsum(net)-np.maximum.accumulate(np.r_[0,np.cumsum(net)])[1:])),
    }
    return {
        'positions':positions,
        'probabilities':sampled_p,
        'forward_log_returns':fwd,
        'gross':gross,
        'costs':cost_paid,
        'net':net,
        'cumulative_net':np.cumsum(net),
        'summary':summary,
    }


def ensure_matches_archived_policy(probabilities, labels, returns, archived, *, horizon=4, atol=1e-9):
    actual=evaluate_policy(probabilities,labels,returns,horizon=horizon,distance=.05,cost=.0005)
    ours=replay(probabilities,returns,horizon=horizon)
    for key in ('nonoverlap_intervals','nonoverlap_active_intervals','turnover_units_including_flatten','net_log_pnl_proxy'):
        ours_key={'nonoverlap_intervals':'n_intervals','nonoverlap_active_intervals':'n_active',
                  'turnover_units_including_flatten':'turnover','net_log_pnl_proxy':'net_log_pnl_proxy'}[key]
        if not np.isclose(float(actual[key]),float(ours['summary'][ours_key]),atol=atol,rtol=0):
            raise RuntimeError('Replay does not reproduce in-repository policy: '+key)
        if not np.isclose(float(archived[key]),float(ours['summary'][ours_key]),atol=atol,rtol=0):
            raise ValueError('Archive and replay disagree: '+key)
    return ours
