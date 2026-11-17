"""Read-only position sizing and SL/TP plan; no external execution path.

Point value MUST be verified for the specific broker CFD contract, volume,
account denomination and conversion. It cannot be inferred from index names.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class RiskRequest:
    balance: float
    risk_percent: float
    entry: float
    stop: float
    take: float
    side: str
    account_value_per_price_point_per_lot: float
    min_lots: float
    lot_step: float
    max_lots: float
    roundtrip_fees_per_lot: float = 0.0
    spread_and_slippage_points: float = 0.0


def _finite_positive(v, name, allow_zero=False):
    if not isinstance(v, (int,float)) or not math.isfinite(v) or (v < 0 if allow_zero else v <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')


def plan(req: RiskRequest) -> dict:
    for k in ('balance','risk_percent','entry','stop','take','account_value_per_price_point_per_lot','min_lots','lot_step','max_lots'):
        _finite_positive(getattr(req,k),k)
    for k in ('roundtrip_fees_per_lot','spread_and_slippage_points'):
        _finite_positive(getattr(req,k),k,True)
    if not (0 < req.risk_percent <= 2):
        raise ValueError('Research/demo risk cap is 2% per trade')
    if req.min_lots > req.max_lots or req.side not in ('buy','sell'):
        raise ValueError('Invalid broker lot bounds or trade side')
    if req.side=='buy' and not (req.stop < req.entry < req.take):
        raise ValueError('Buy plans require stop below entry and take above entry')
    if req.side=='sell' and not (req.take < req.entry < req.stop):
        raise ValueError('Sell plans require take below entry and stop above entry')
    sl=abs(req.entry - req.stop)
    tp=abs(req.take - req.entry)
    allowance=req.balance*req.risk_percent/100
    risk_per_lot=(sl+req.spread_and_slippage_points)*req.account_value_per_price_point_per_lot+req.roundtrip_fees_per_lot
    gain_per_lot=max(0.0,(tp-req.spread_and_slippage_points)*req.account_value_per_price_point_per_lot-req.roundtrip_fees_per_lot)
    if risk_per_lot <= 0:
        raise ValueError('Invalid zero-risk contract')
    # Round down to the broker volume grid starting at min_lots, never round up.
    n=math.floor((min(req.max_lots,allowance/risk_per_lot)-req.min_lots)/req.lot_step + 1e-10)
    if n < 0:
        return {'status':'no_trade','reason':'Minimum broker lot exceeds account risk cap',
                'risk_budget':round(allowance,6),'volume_lots':0.0,'max_loss':0.0,
                'risk_reward_net':None,'execution_enabled':False}
    lots=req.min_lots+n*req.lot_step
    if lots*risk_per_lot > allowance+1e-9:
        raise ValueError('Volume violates risk budget')
    return {'status':'manual_plan_only','side':req.side,'entry':req.entry,'stop_loss':req.stop,
            'take_profit':req.take,'volume_lots':round(lots,8),
            'risk_budget':round(allowance,6), 'max_loss':round(lots*risk_per_lot,6),
            'estimated_gain':round(lots*gain_per_lot,6),
            'risk_reward_gross':round(tp/sl,4),
            'risk_reward_net':round(gain_per_lot/risk_per_lot,4),
            'estimated_risk_percent_of_balance':round(100*lots*risk_per_lot/req.balance,5),
            'execution_enabled':False,
            'warning':'Illustrative price-based risk. No guarantee of stop fill; verify tick value, contract size, currency conversion, minimum stop distance, commissions, and margin with Deriv MT5 before any order.'}
