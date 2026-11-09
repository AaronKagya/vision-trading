"""Forward-only BTCUSDT spot paper monitor, **not** a trained AI or brokerage.

A scheduled observer uses public 15m completed candles for a fixed one-hour
momentum rule, and the public last-traded ticker *as seen on the observation
run* for hypothetical entry/exit prices. No backfilling, leverage, shorts,
orders, credentials, exchange accounts, or invented predictions.

The long-only strategy is an unvalidated baseline. The 0.0005-per-side cost
is illustrative; returned values are simulated directional log-P&L proxies.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import urllib.parse
import urllib.request
from typing import Any

INTERVAL_MS = 15 * 60 * 1000
HOUR_MS = 4 * INTERVAL_MS
COST = 0.0005
BASE_URL = 'https://data-api.binance.vision'
RULE = 'btc_spot_momentum4_long_flat_baseline_v1'
SCHEMA = 1
MAX_SIGNAL_DELAY_MS = 12 * 60 * 1000
MAX_PRICE_AGE_MS = 30 * 60 * 1000


def iso(millis: int) -> str:
    return dt.datetime.fromtimestamp(int(millis) / 1000, tz=dt.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _positive(value: Any) -> float:
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise ValueError('Invalid price')
    return x


def get_public_market_snapshot(*, timeout: int = 18) -> tuple[list, float, int]:
    """Use only Binance's publicly documented market-data-only endpoints."""
    query = urllib.parse.urlencode({'symbol': 'BTCUSDT', 'interval': '15m', 'limit': 160})
    req = urllib.request.Request(BASE_URL + '/api/v3/klines?' + query, headers={'User-Agent': 'VisionTradingResearch/0.4'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = json.load(response)
    req2 = urllib.request.Request(BASE_URL + '/api/v3/ticker/price?symbol=BTCUSDT', headers={'User-Agent': 'VisionTradingResearch/0.4'})
    with urllib.request.urlopen(req2, timeout=timeout) as response:
        ticker = json.load(response)
    if not isinstance(ticker, dict) or ticker.get('symbol') != 'BTCUSDT':
        raise ValueError('Unexpected ticker symbol')
    price = _positive(ticker['price'])
    if not isinstance(raw, list):
        raise ValueError('Unexpected candle response')
    return raw, price, int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)


def closed_candles(raw: list, observed_ms: int) -> list[dict]:
    """Validate timestamps, OHLC and gaps; discard incomplete 15m candle."""
    if not isinstance(raw, list) or len(raw) < 10:
        raise ValueError('Too few candle rows')
    seen = []
    for row in raw:
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError('Malformed kline response')
        stamp = int(row[0])
        if stamp % INTERVAL_MS:
            raise ValueError('Misaligned 15-minute timestamp')
        op, high, low, close = map(_positive, row[1:5])
        vol = float(row[5])
        if not math.isfinite(vol) or vol < 0 or high < max(op, close, low) or low > min(op, close, high):
            raise ValueError('Invalid OHLCV')
        seen.append({'open_ms': stamp, 'close_ms': stamp + INTERVAL_MS,
                     'open': op, 'high': high, 'low': low, 'close': close})
    for old, new in zip(seen, seen[1:]):
        if new['open_ms'] - old['open_ms'] != INTERVAL_MS:
            raise ValueError('Duplicate, missing, or unordered 15-minute candles')
    completed = [c for c in seen if c['close_ms'] <= observed_ms - 3000]
    if len(completed) < 9 or completed[-1]['close_ms'] + MAX_PRICE_AGE_MS < observed_ms:
        raise ValueError('Stale or incomplete market snapshot')
    return completed


def new_state(observed_ms: int) -> dict:
    return {'schema_version': SCHEMA, 'symbol': 'BTCUSDT', 'venue': 'Binance spot',
            'strategy': RULE, 'strategy_type': 'non_ai_baseline',
            'activated_at_ms': observed_ms, 'updated_at_ms': observed_ms,
            'last_signal_hour_ms': None, 'open_trade': None, 'ledger': [],
            'market': None, 'status': 'monitoring', 'transaction_cost_per_side': COST,
            'notes': 'Hypothetical paper observations, not trades or model forecasts.'}


def _check_state(state: dict) -> None:
    if state.get('schema_version') != SCHEMA or state.get('symbol') != 'BTCUSDT' or state.get('strategy') != RULE:
        raise ValueError('State schema/strategy mismatch: refusing to change an active ledger')
    if state.get('transaction_cost_per_side') != COST:
        raise ValueError('Cost configuration changed')
    if not isinstance(state.get('ledger'), list) or not isinstance(state.get('activated_at_ms'), int):
        raise ValueError('Malformed ledger state')


def advance(state: dict | None, raw: list, ticker_price: float, observed_ms: int) -> dict:
    """Advance one observed event. Historical candles may inform a signal; entries never predate activation.

    Calling the function twice for the same observed hour does not duplicate a
    decision. Missed signal windows are deliberately NOT backfilled.
    The hypothetical entry or exit is priced using the ticker observed *now*,
    never the signal candle's earlier closing price.
    """
    import copy
    ticker_price = _positive(ticker_price)
    completed = closed_candles(raw, observed_ms)
    if state is None:
        next_state = new_state(observed_ms)
    else:
        _check_state(state)
        next_state = copy.deepcopy(state)
        if observed_ms < state['updated_at_ms']:
            raise ValueError('Time regression; state not mutated')
    last = completed[-1]
    next_state['updated_at_ms'] = observed_ms
    next_state['market'] = {
        'observed_at_ms': observed_ms,
        'closed_candle_at_ms': last['close_ms'],
        'last_trade_price_at_observation': ticker_price,
        'last_candle_close': last['close'],
        'last_96_closes': [[c['close_ms'], c['close']] for c in completed[-96:]],
        'source': 'Binance public 15m klines + BTCUSDT spot ticker (unauthenticated)',
    }

    # Only settle from a *new market observation* and at least one hour after
    # entry. Settlement is at the latest public ticker price at that observed time.
    opened = next_state['open_trade']
    if opened and observed_ms >= opened['entry_observed_ms'] + HOUR_MS:
        pnl_gross = math.log(ticker_price / opened['entry_price'])
        pnl_cost = 2 * COST
        pos = next((x for x in next_state['ledger'] if x.get('id') == opened['id']), None)
        if pos is None or pos['status'] != 'open':
            raise ValueError('Open position missing from ledger')
        pos.update({'status': 'closed', 'exit_observed_ms': observed_ms,
                    'exit_price': ticker_price, 'held_minutes': round((observed_ms - opened['entry_observed_ms'])/60000, 2),
                    'gross_log_pnl_proxy': pnl_gross, 'cost_log_pnl_proxy': pnl_cost,
                    'net_log_pnl_proxy': pnl_gross - pnl_cost})
        next_state['open_trade'] = None

    # A signal is eligible once every hour, on the candle closing at :00 UTC;
    # only while it is fresh and was unavailable at activation time.
    candle_time = last['close_ms']
    eligible = (candle_time % HOUR_MS == 0
                and 0 <= observed_ms - candle_time <= MAX_SIGNAL_DELAY_MS
                and candle_time > next_state['activated_at_ms']
                and (next_state['last_signal_hour_ms'] is None or candle_time > next_state['last_signal_hour_ms']))
    if eligible:
        current = last['close']
        prior = completed[-5]['close']  # four completed 15m returns, all past
        momentum = current / prior - 1
        action = 'long' if momentum > 0 else 'flat'
        # If a previous paper observation is still open due to missed polls,
        # never overlap an assumed one-hour position.
        if action == 'long' and next_state['open_trade']:
            action = 'flat'
        identity = 'BTCUSDT-' + iso(candle_time)
        if any(entry['id'] == identity for entry in next_state['ledger']):
            raise ValueError('Duplicate decision identity detected')
        record = {
            'id': identity, 'signal_candle_close_ms': candle_time,
            'signal_observed_ms': observed_ms, 'signal_candle_close': current,
            'one_hour_prior_close': prior, 'momentum_1h': momentum,
            'action': action, 'status': 'open' if action == 'long' else 'flat',
            'entry_observed_ms': observed_ms if action == 'long' else None,
            'entry_price': ticker_price if action == 'long' else None,
            'exit_observed_ms': None, 'exit_price': None,
            'gross_log_pnl_proxy': None if action == 'long' else 0.0,
            'cost_log_pnl_proxy': None if action == 'long' else 0.0,
            'net_log_pnl_proxy': None if action == 'long' else 0.0,
            'signal_type': 'momentum_baseline_not_ai',
        }
        next_state['ledger'].append(record)
        if action == 'long':
            next_state['open_trade'] = {'id': identity, 'entry_observed_ms': observed_ms,
                                        'entry_price': ticker_price}
        next_state['last_signal_hour_ms'] = candle_time

    next_state['status'] = 'paper_long_open' if next_state['open_trade'] else 'monitoring_flat'
    return next_state


def public_summary(state: dict) -> dict:
    """Public, credential-free JSON. Show only observed prices and simulated data."""
    _check_state(state)
    settled = [x for x in state['ledger'] if x['status'] == 'closed']
    net = sum(x['net_log_pnl_proxy'] for x in settled)
    return {**state, 'metrics': {
        'decisions': len(state['ledger']), 'closed_longs': len(settled),
        'no_trade_decisions': sum(x['status'] == 'flat' for x in state['ledger']),
        'net_log_pnl_proxy': net, 'illustrative_cost_assumption_per_side': COST,
    }}
