"""Prospective *observation-only* LSTM shadow evaluation on BTCUSDT spot.

Five frozen 2024 validation-selected (20 maximum epoch) checkpoints are an
ensemble. No online fitting, broker connectivity, orders or trade advice.
Each forecast is committed only after a new candle closes, and its label is
populated only after a later completed candle becomes observable.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

SYMBOL = 'BTCUSDT'
INTERVAL_MS = 15 * 60_000
HORIZON_MS = 4 * INTERVAL_MS
MAX_DELAY_MS = 12 * 60_000
STALE_MS = 30 * 60_000
EXPECTED_DATA_HASH = '2cc72f14a30e03baa1aeff87d4d6c63b4359361146fa51b633b96a21e106de67'
BASE_URL = 'https://data-api.binance.vision'
SEEDS = tuple(range(5))
SCHEMA = 1
MODEL = 'frozen_2024_lstm_five_seed_equal_mean_v1'


def iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def checkpoint_paths(root='models/phase4_2024') -> list[Path]:
    return [Path(root) / f'lstm_seed{s}.pt' for s in SEEDS]


def verified_checkpoint_manifest(root='models/phase4_2024') -> dict:
    """Compute a stable identity for the exact five-ensemble immutable weights."""
    paths = checkpoint_paths(root)
    sha = {}
    for seed, path in zip(SEEDS, paths):
        if not path.is_file():
            raise FileNotFoundError(f'Missing frozen seed {seed}: {path}')
        sha[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'model_id': MODEL, 'weights_sha256_by_seed': sha,
            'training_data_sha256': EXPECTED_DATA_HASH,
            'training_period': '2024 UTC; fit before 2024-10-01; validation through 2024-12-31',
            'validation_selected_max_epochs': 20,
            'ensemble_policy': 'unweighted mean of all five fixed seeds',
            'forecast_horizon_candles': 4, 'forecast_timeframe': '15m'}


def verified_models(root='models/phase4_2024'):
    """Load trusted research-artifact tensor weights with restricted deserialization."""
    import torch
    from src.models import LSTMBaseline

    result = []
    for seed, path in zip(SEEDS, checkpoint_paths(root)):
        pack = torch.load(path, map_location='cpu', weights_only=True)
        config = pack.get('config', {})
        if (pack.get('data_sha256_2024') != EXPECTED_DATA_HASH
            or pack.get('model') != 'LSTMBaseline'
            or config.get('seed') != seed or config.get('window') != 32
            or config.get('horizon') != 4):
            raise ValueError(f'Unrecognized seed {seed} 2024 model provenance')
        scaler = pack.get('scaler_std')
        if scaler is None or scaler.shape != (12,) or not bool(torch.isfinite(scaler).all()) or not bool((scaler > 0).all()):
            raise ValueError(f'Invalid model scaler for seed {seed}')
        model = LSTMBaseline(12).cpu()
        model.load_state_dict(pack['state_dict'], strict=True)
        model.eval()
        result.append((model, scaler))
    return result


def predict_frozen_ensemble(history: pd.DataFrame, models) -> tuple[float, list[float]]:
    """Compute from *completed* trailing OHLC only, using 2024 train scalers."""
    import torch
    from src.features import panels, numeric_features

    _, pan = panels(history, window=32)
    x = numeric_features(pan[-1:])  # shape 1 x 32 x 12, ending at last closed candle
    out = []
    with torch.inference_mode():
        for model, scaler in models:
            pred = model(torch.as_tensor(x, dtype=torch.float32) / scaler).sigmoid().item()
            if not math.isfinite(pred) or not 0 <= pred <= 1:
                raise ValueError('Nonfinite model output')
            out.append(float(pred))
    return float(np.mean(out)), out


def get_public_klines(*, timeout: int = 25, observed_ms: int | None = None) -> tuple[pd.DataFrame, int]:
    """Fetch 720 consecutive OHLCV rows and discard the still-open candle."""
    query = urllib.parse.urlencode({'symbol': SYMBOL, 'interval': '15m', 'limit': 720})
    req = urllib.request.Request(BASE_URL + '/api/v3/klines?' + query,
                                 headers={'User-Agent': 'VisionTradingResearch/0.6'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = json.load(response)
    now_ms = observed_ms if observed_ms is not None else int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    return validated_closed_klines(raw, now_ms), now_ms


def validated_closed_klines(raw, observed_ms: int) -> pd.DataFrame:
    if not isinstance(raw, list) or len(raw) < 520:
        raise ValueError('Fewer than 520 klines: cannot construct a 32×4h panel')
    candles = []
    prev = None
    for row in raw:
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError('Malformed candle')
        stamp = int(row[0])
        if stamp % INTERVAL_MS or (prev is not None and stamp - prev != INTERVAL_MS):
            raise ValueError('Missing, repeated or unaligned 15m candle')
        prev = stamp
        values = tuple(map(float, row[1:6]))
        o,h,l,c,v = values
        if (not all(math.isfinite(t) for t in values) or min(o,h,l,c) <= 0
            or v < 0 or h < max(o,c,l) or l > min(o,c,h)):
            raise ValueError('Invalid OHLCV')
        if stamp + INTERVAL_MS <= observed_ms - 3000:
            candles.append([stamp,o,h,l,c,v])
    if len(candles) < 520:
        raise ValueError('Not enough completed candles for temporal features')
    if observed_ms < candles[-1][0]+INTERVAL_MS or observed_ms - (candles[-1][0]+INTERVAL_MS) > STALE_MS:
        raise ValueError('Stale or future market data')
    df=pd.DataFrame(candles,columns=['open_ms','open','high','low','close','volume'])
    df['time']=pd.to_datetime(df.open_ms,unit='ms',utc=True)
    return df


def new_state(now_ms: int, manifest: dict) -> dict:
    return {'schema_version': SCHEMA, 'symbol': SYMBOL, 'strategy_type': 'ai_forecast_shadow_no_orders',
            'model': manifest, 'activated_at_ms': now_ms, 'updated_at_ms': now_ms,
            'last_forecast_close_ms': None, 'market': None, 'forecasts': [],
            'status': 'armed_no_prior_predictions'}


def _check_state(state: dict, manifest: dict):
    if (state.get('schema_version') != SCHEMA or state.get('symbol') != SYMBOL
        or state.get('strategy_type') != 'ai_forecast_shadow_no_orders'
        or state.get('model') != manifest):
        raise ValueError('Model version mismatch; refuse to rewrite an existing forecast ledger')
    if not isinstance(state.get('forecasts'), list) or not isinstance(state.get('activated_at_ms'), int):
        raise ValueError('Malformed shadow ledger')


def advance(state: dict | None, history: pd.DataFrame, now_ms: int,
            manifest: dict, predict=None) -> dict:
    """Immutable advance; no retrospective forecast fabrication or future-candle input."""
    if not isinstance(now_ms,int) or history.empty:
        raise ValueError('Invalid observation')
    history = history.copy()
    last = history.iloc[-1]
    last_close_ms = int(last.open_ms) + INTERVAL_MS
    if now_ms < last_close_ms or now_ms - last_close_ms > STALE_MS:
        raise ValueError('Market observation stale or future')
    if len(history) < 520 or len(history.open_ms) > 1 and (history.open_ms.diff().iloc[1:] != INTERVAL_MS).any():
        raise ValueError('Incomplete historical panel')
    if state is None:
        result = new_state(now_ms, manifest)
    else:
        _check_state(state, manifest)
        if now_ms < state['updated_at_ms']:
            raise ValueError('Observation time regression')
        result=copy.deepcopy(state)
    result['updated_at_ms'] = now_ms
    result['market'] = {'observed_at_ms':now_ms,'closed_at_ms':last_close_ms,
                        'last_closed_price':float(last.close),
                        'source':'Binance public BTCUSDT spot completed 15m klines'}
    closes={int(r.open_ms)+INTERVAL_MS: float(r.close) for r in history.itertuples(index=False)}
    for entry in result['forecasts']:
        if entry['status'] != 'pending':
            continue
        target=entry['target_close_ms']
        if target in closes and target <= last_close_ms:
            actual=closes[target]
            y=int(actual > entry['origin_price'])
            entry.update({'status':'settled','outcome_price':actual,'up':y,
                          'model_correct':int(entry['prob_up']>=.5)==y,
                          'baseline_correct':entry['baseline_up']==y,
                          'label_observed_at_ms':now_ms})
        elif last_close_ms > target and target < min(closes):
            entry['status']='outcome_unavailable'
    eligible=(state is not None and last_close_ms > result['activated_at_ms']
              and last_close_ms % HORIZON_MS == 0
              and 0 <= now_ms - last_close_ms <= MAX_DELAY_MS
              and (result['last_forecast_close_ms'] is None or
                   last_close_ms > result['last_forecast_close_ms']))
    if eligible:
        if predict is None:
            raise ValueError('Inference callback required for eligible new forecasts')
        if len(history)<5:
            raise ValueError('Not enough history to calculate baseline')
        p, per_seed = predict(history)
        if not math.isfinite(p) or not 0 <= p <= 1 or len(per_seed)!=5 or any(not math.isfinite(x) or x < 0 or x > 1 for x in per_seed):
            raise ValueError('Bad forecast output')
        if abs(p-np.mean(per_seed)) > 1e-8:
            raise ValueError('Ensemble value differs from five seed average')
        ident=f'{SYMBOL}_{iso(last_close_ms)}'
        if any(e['id']==ident for e in result['forecasts']):
            raise ValueError('Duplicate forecast identity')
        result['forecasts'].append({
            'id':ident,'origin_close_ms':last_close_ms,'forecast_observed_ms':now_ms,
            'origin_price':float(last.close),'target_close_ms':last_close_ms + HORIZON_MS,
            'horizon_candles':4,'prob_up':round(float(p),9),
            'prob_up_by_seed':[round(float(x),9) for x in per_seed],
            'baseline_up':int(float(last.close)>float(history.iloc[-5].close)),
            'status':'pending','outcome_price':None,'up':None,
            'model_correct':None,'baseline_correct':None,'label_observed_at_ms':None,
            'signal_type':'model_observation_not_trade',
        })
        result['last_forecast_close_ms']=last_close_ms
    result['status']='monitoring' if result['forecasts'] else 'armed_no_prior_predictions'
    return result


def public_summary(state: dict) -> dict:
    _check_state(state, state['model'])
    settled=[v for v in state['forecasts'] if v['status']=='settled']
    n=len(settled)
    metrics={
        'forecasts':len(state['forecasts']), 'settled':n,
        'pending':sum(v['status']=='pending' for v in state['forecasts']),
        'missing_outcomes':sum(v['status']=='outcome_unavailable' for v in state['forecasts']),
        'accuracy':sum(x['model_correct'] for x in settled)/n if n else None,
        'momentum_baseline_accuracy':sum(x['baseline_correct'] for x in settled)/n if n else None,
        'brier_score':sum((x['prob_up']-x['up'])**2 for x in settled)/n if n else None,
    }
    return {**state,'metrics':metrics,'research_disclaimer':'Frozen 2024 LSTM ensemble; forward recorded forecasts only. No calibration guarantee, trading orders, or profitability claim.'}
