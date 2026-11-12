"""Download and strictly verify Binance SPOT monthly 15-minute candles.

Only BTCUSDT/ETHUSDT/SOLUSDT are currently approved for the comparison.
No account credentials, live trading, or broker execution are used.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import pandas as pd

from src.data import download_binance, load_candles

ALLOWED = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_period(df, start_month, end_month):
    start = pd.Period(start_month, freq='M')
    end = pd.Period(end_month, freq='M')
    if end < start:
        raise ValueError('End month cannot precede start month')
    begin = pd.Timestamp(start.start_time, tz='UTC')
    finish = pd.Timestamp((end + 1).start_time, tz='UTC')
    expected = int((finish - begin) / pd.Timedelta(minutes=15))
    if (str(df.time.dt.tz) != 'UTC' or len(df) != expected
        or df.time.iloc[0] != begin or df.time.iloc[-1] != finish - pd.Timedelta(minutes=15)
        or (df.time.diff().iloc[1:] != pd.Timedelta(minutes=15)).any()):
        raise ValueError(f'Incomplete or irregular 15m candles for {start_month} to {end_month}')
    return {'candles':expected,'first_candle_utc':begin.isoformat(),
            'last_candle_utc':df.time.iloc[-1].isoformat(),'interval':'15m'}


def prepare(symbol: str, start='2024-01', end='2024-12', out_dir='data'):
    symbol = str(symbol).upper()
    if symbol not in ALLOWED:
        raise ValueError(f'Unapproved instrument {symbol!r}; choose from {ALLOWED}')
    file = Path(download_binance(symbol, '15m', start, end, out_dir=out_dir))
    df = load_candles(file, strict=True)
    period = validate_period(df, start, end)
    manifest = {
        'instrument': symbol, 'venue': 'Binance spot', 'timeframe': '15m',
        'market_data_type': 'OHLCV', 'source': 'https://data.binance.vision',
        'months_inclusive': [start, end], 'file_name': file.name,
        'sha256': sha256(file), 'rows': len(df), **period,
        'note': 'Historical market data only; no orders or live performance claims.'
    }
    target = file.with_suffix('.manifest.json')
    target.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'{symbol}: {len(df):,} verified 15m candles, sha256={manifest["sha256"]}', flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--symbol', choices=ALLOWED, required=True)
    p.add_argument('--start', default='2024-01')
    p.add_argument('--end', default='2024-12')
    p.add_argument('--out-dir', default='data')
    a = p.parse_args()
    prepare(a.symbol, a.start, a.end, a.out_dir)


if __name__ == '__main__':
    main()
