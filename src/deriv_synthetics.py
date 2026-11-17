"""Deriv synthetic-index PUBLIC market-data adapter; never authenticates or trades.

Only quote history and instrument identity available from public active_symbols.
No hard-coded broker symbol codes: discovered for each requested instrument.
A fixed volatility setting does NOT imply directional predictability.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data import validate_candles

PUBLIC_WS = 'wss://api.derivws.com/trading/v1/options/ws/public'
GRANULARITY = 900
TRACKED_NAMES = (
    'Volatility 10 Index', 'Volatility 25 Index', 'Volatility 75 Index',
    'Jump 50 Index', 'Jump 75 Index', 'Jump 100 Index',
)
# Exact names distinguish traditional 2s Volatility Indices from 1s variants.
_NORMALIZED = {re.sub(r'\s+', ' ', k).lower(): k for k in TRACKED_NAMES}
_SYMBOL = re.compile(r'^\w{2,30}$')


def discover(symbols: list[dict]) -> dict[str, dict]:
    """Map broker-provided display names to current symbol IDs; never guess IDs."""
    if not isinstance(symbols, list):
        raise ValueError('active_symbols response is not a list')
    seen: dict[str, dict] = {}
    for item in symbols:
        if not isinstance(item, dict):
            raise ValueError('Invalid symbol metadata')
        name = re.sub(r'\s+', ' ', str(item.get('display_name', '')).strip())
        canonical = _NORMALIZED.get(name.lower())
        if canonical is None:
            continue
        code = item.get('symbol')
        if not isinstance(code, str) or not _SYMBOL.fullmatch(code):
            raise ValueError(f'Invalid API symbol for {canonical}')
        if canonical in seen and seen[canonical]['symbol'] != code:
            raise ValueError(f'Ambiguous API symbol for {canonical}')
        seen[canonical] = {'symbol': code, 'display_name': name,
                           'market': item.get('market'), 'submarket': item.get('submarket')}
    return seen


def parse_candles(message: dict) -> list[dict]:
    """Validate candle payload; open timestamp is seconds UTC, not close time."""
    if not isinstance(message, dict) or message.get('error'):
        raise ValueError(f'Deriv market-data error: {str(message.get("error"))[:200]}')
    if message.get('msg_type') != 'candles' or not isinstance(message.get('candles'), list):
        raise ValueError('Expected candles market-data response')
    data = []
    for row in message['candles']:
        if not isinstance(row, dict):
            raise ValueError('Malformed historical candle')
        try:
            stamp = int(row['epoch'])
            o,h,l,c = (float(row[x]) for x in ('open', 'high', 'low', 'close'))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError('Unparseable historical candle') from exc
        if stamp % GRANULARITY or not all(math.isfinite(v) and v > 0 for v in (o,h,l,c)):
            raise ValueError('Bad timestamps or nonfinite/nonpositive prices')
        if h < max(o,l,c) or l > min(o,h,c):
            raise ValueError('Inconsistent candle OHLC')
        data.append({'epoch': stamp, 'open': o, 'high': h, 'low': l, 'close': c})
    if not data:
        raise ValueError('Empty candle page')
    stamps=[x['epoch'] for x in data]
    if stamps != sorted(set(stamps)):
        raise ValueError('Candles out of order or duplicate')
    return data


def validate_history(rows: list[dict], *, observed_epoch: int) -> pd.DataFrame:
    if not rows:
        raise ValueError('No market history')
    rows = sorted(rows, key=lambda x: x['epoch'])
    times = [r['epoch'] for r in rows]
    if len(set(times)) != len(times) or any(b-a != GRANULARITY for a,b in zip(times, times[1:])):
        raise ValueError('Candle gaps, duplicates, or mixed sampling')
    # The last / partly forming bar cannot be used in research features.
    closed = [r for r in rows if r['epoch'] + GRANULARITY <= observed_epoch - 3]
    if len(closed) < 520:
        raise ValueError('Less than 520 complete candles; insufficient for multiscale panels')
    if observed_epoch - closed[-1]['epoch'] - GRANULARITY > 1800:
        raise ValueError('Latest candle is stale')
    frame = pd.DataFrame(closed)
    frame['time'] = pd.to_datetime(frame.epoch, unit='s', utc=True)
    frame['volume'] = 0.0  # Explicit placeholder: Deriv candle endpoint does NOT supply trade volume.
    return validate_candles(frame[['time','open','high','low','close','volume']], strict=True)


def fetch_public(request: dict, *, timeout: int = 20) -> dict:
    """One authenticated-free public request per connection; fail on upstream errors."""
    from websockets.sync.client import connect
    if any(k in request for k in ('buy', 'sell', 'authorize', 'proposal', 'contract_update')):
        raise ValueError('Trading requests are prohibited in public market-data adapter')
    with connect(PUBLIC_WS, open_timeout=timeout, close_timeout=5) as ws:
        ws.send(json.dumps({**request, 'req_id': 12345}))
        msg = json.loads(ws.recv(timeout=timeout))
    if not isinstance(msg, dict) or msg.get('error') or msg.get('errors'):
        raise RuntimeError('Deriv API error: '+str(msg.get('error') or msg.get('errors'))[:240])
    return msg


def get_symbols(fetch: Callable[[dict],dict] = fetch_public) -> dict[str,dict]:
    response = fetch({'active_symbols':'brief'})
    if response.get('msg_type') != 'active_symbols':
        raise ValueError('Unexpected active_symbols response type')
    return discover(response.get('active_symbols'))


def download_recent(name: str, *, pages: int = 4,
                    fetch: Callable[[dict],dict] = fetch_public,
                    now_epoch: int | None = None) -> tuple[pd.DataFrame,dict]:
    """Page 15-minute candles backwards. Never backfill with simulated data.

    Returns a causally usable CSV-compatible frame and recorded provenance.
    """
    if name not in TRACKED_NAMES:
        raise ValueError('Instrument outside approved research universe')
    if not isinstance(pages,int) or not 1 <= pages <= 40:
        raise ValueError('pages must be 1..40')
    available = get_symbols(fetch)
    if name not in available:
        raise ValueError(f'{name} not returned by active_symbols; no invented symbol code')
    code=available[name]['symbol']
    end: str = 'latest'
    observed = int(now_epoch if now_epoch is not None else datetime.now(timezone.utc).timestamp())
    records=[]
    for _ in range(pages):
        response=fetch({'ticks_history':code,'end':end,'count':1000,
                        'style':'candles','granularity':GRANULARITY})
        chunk=parse_candles(response)
        records = chunk + records
        end=str(chunk[0]['epoch'] - 1)
    df=validate_history(records, observed_epoch=observed)
    manifest={'provider':'Deriv','data_type':'public historical candle observations',
              'market_kind':'algorithmic synthetic index','display_name':name,'symbol':code,
              'timeframe_seconds':GRANULARITY,'timezone':'UTC',
              'start_utc':df.time.iloc[0].isoformat(), 'end_utc':df.time.iloc[-1].isoformat(),
              'candles':len(df),'pages_requested':pages,
              'volume_note':'API candle history has no traded volume; column filled with zero and excluded from predictive features',
              'source':PUBLIC_WS,'data_claim':'observations only; no predictive skill claimed'}
    return df,manifest


def save(df: pd.DataFrame, manifest: dict, folder: str | Path):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    name=re.sub(r'[^A-Za-z0-9]+','_',manifest['display_name']).strip('_').lower()
    csv=folder/f'deriv_{name}_15m.csv'
    df.to_csv(csv,index=False)
    digest=hashlib.sha256(csv.read_bytes()).hexdigest()
    manifest={**manifest,'sha256_csv':digest,'csv_file':csv.name}
    path=csv.with_suffix('.manifest.json')
    path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return csv,path
