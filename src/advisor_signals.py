"""Fail-closed evidence gate for external trading-advisor API signals.

A provider's advertising or backtest win-rate claim is never substituted for
independently recorded outcomes. This module has NO account credentials and
NO buy/sell transport. A research UI may label signals unverified and withhold
execution unless prospective statistics and product terms are reviewed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSignal:
    provider: str
    instrument: str
    model_version: str
    issued_utc_epoch: int
    horizon_seconds: int
    direction: str
    probability_up: float
    origin_data_epoch: int


def validate_candidate(signal: CandidateSignal, *, observed_epoch: int) -> dict:
    if not signal.provider or not signal.model_version or not signal.instrument:
        raise ValueError('Missing provider/model/instrument provenance')
    if signal.direction not in ('up','down','abstain'):
        raise ValueError('Unsupported directional signal')
    if not isinstance(signal.probability_up,(float,int)) or not math.isfinite(signal.probability_up) or not 0<=signal.probability_up<=1:
        raise ValueError('Probability must be finite and in [0,1]')
    if signal.issued_utc_epoch > observed_epoch or signal.origin_data_epoch > signal.issued_utc_epoch:
        raise ValueError('Future data or future-issued signal')
    if signal.horizon_seconds <= 0 or observed_epoch-signal.issued_utc_epoch > 300:
        raise ValueError('Stale signal or unsupported horizon')
    return {'status':'research_candidate_unverified','provider':signal.provider,
            'instrument':signal.instrument,'model_version':signal.model_version,
            'direction':signal.direction,'probability_up_uncalibrated':signal.probability_up,
            'issued_utc_epoch':signal.issued_utc_epoch,'valid_for_seconds':signal.horizon_seconds,
            'allow_orders':False,
            'required_evidence':'Time-stamped forward forecasts, paired realized outcomes, independent validation, costs and calibration; explicit permitted API use'}
