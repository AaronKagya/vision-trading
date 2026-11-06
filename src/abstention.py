"""Predeclared abstention policy and honest historical execution *proxies*.

IMPORTANT: Threshold 0.05 (P(up) >= .55 or <= .45) is a predeclared
research hypothesis, not an optimized trading rule or calibrated probability.
Historical performance at this threshold is exploratory when evaluated on
2024; seek independent future-date validation before drawing conclusions.
"""
from __future__ import annotations
import numpy as np

DEFAULT_DISTANCE = 0.05


def positions_from_probabilities(probabilities, distance=DEFAULT_DISTANCE, *, long_only=False):
    """Position in {-1,0,+1}; only inputs available at the prediction time."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be a finite one-dimensional array in [0,1].')
    if not np.isfinite(distance) or not (0 <= distance <= .5):
        raise ValueError('Distance must be between zero and 0.5.')
    # Strictly greater than threshold prevents meaningless 50/50 calls.
    if distance == 0:
        return np.where(p > .5, 1, np.where(p < .5, -1 if not long_only else 0, 0)).astype(np.int8)
    hi = p >= .5 + distance
    lo = p <= .5 - distance
    return np.where(hi, 1, np.where(lo & (not long_only), -1, 0)).astype(np.int8)


def evaluate_policy(probabilities, labels, forward_log_returns, *, horizon=4,
                    distance=DEFAULT_DISTANCE, cost=.0005, long_only=False):
    """Accuracy/coverage across all observations; log-P&L on disjoint intervals.

    This is NOT a broker backtest: no book depth, bid/ask, filling, slippage,
    funding, borrow, margin, liquidations, latency, or mark-to-market.
    We charge a cost on both entry and terminal liquidation. Starting flat.
    """
    p = np.asarray(probabilities, float)
    y = np.asarray(labels)
    fwd = np.asarray(forward_log_returns, float)
    if p.ndim != 1 or y.shape != p.shape or fwd.shape != p.shape or not len(p):
        raise ValueError('Arrays must be nonempty, one-dimensional, same length.')
    if not np.isfinite(fwd).all() or not np.isin(y, [0,1]).all():
        raise ValueError('Invalid labels or returns.')
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('horizon must be positive integer.')
    if not np.isfinite(cost) or cost < 0:
        raise ValueError('cost must be nonnegative and finite.')
    pos = positions_from_probabilities(p,distance,long_only=long_only)
    selected = pos != 0
    idx = np.arange(0,len(p),horizon)
    sampled = pos[idx]
    # Transaction cost per unit exposure change; final flatten counts as cost.
    turnover = np.abs(np.diff(np.r_[0,sampled,0])).sum()
    pnl = float(np.dot(sampled.astype(float),fwd[idx]) - cost * turnover)
    n_selected = int(selected.sum())
    tradable = sampled != 0
    rets = sampled.astype(float) * fwd[idx] - cost*np.abs(np.diff(np.r_[0,sampled]))
    if len(rets):
        rets[-1] -= cost * abs(sampled[-1])
    return {
        'threshold_distance':float(distance),'long_only':bool(long_only),
        'observations':int(len(p)), 'selected_observations':n_selected,
        'coverage':float(selected.mean()),
        'selected_direction_accuracy':float(((p[selected] >= .5)==y[selected]).mean()) if n_selected else None,
        'nonoverlap_intervals':int(len(idx)),
        'nonoverlap_active_intervals':int(tradable.sum()),
        'nonoverlap_active_fraction':float(tradable.mean()),
        'turnover_units_including_flatten':float(turnover),
        'net_log_pnl_proxy':pnl,
        'net_proxy_per_interval_mean':float(rets.mean()),
        'net_proxy_per_interval_std':float(rets.std(ddof=1)) if len(rets)>1 else 0.,
        'note':'Research-only directional log-P&L proxy. Not realized or executable trading returns.'
    }
