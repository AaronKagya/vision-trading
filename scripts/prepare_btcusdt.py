"""Acquire, validate and fingerprint historical BTCUSDT 15m candles.

This script accesses only Binance's public SPOT archive; it does not trade or
request exchange credentials. The raw data is not committed to source control.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data import download_binance, load_candles


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_period(df: pd.DataFrame, start_month: str, end_month: str) -> dict:
    """Fail closed for missing or extra candles, including the period's edges."""
    start = pd.Period(start_month, freq='M')
    end = pd.Period(end_month, freq='M')
    if end < start:
        raise ValueError('End month cannot precede start month')
    begin = pd.Timestamp(start.start_time, tz='UTC')
    finish = pd.Timestamp((end + 1).start_time, tz='UTC')
    if str(df.time.dt.tz) != 'UTC':
        raise ValueError('Timestamps must be UTC')
    expected = int((finish - begin) / pd.Timedelta(minutes=15))
    if len(df) != expected:
        raise ValueError(f'Expected {expected} candles for entire date range; found {len(df)}')
    if df.time.iloc[0] != begin or df.time.iloc[-1] != finish - pd.Timedelta(minutes=15):
        raise ValueError('Candle times do not cover the expected UTC start and end')
    if (df.time.diff().iloc[1:] != pd.Timedelta(minutes=15)).any():
        raise ValueError('Discontinuous 15-minute timestamps')
    return {'candles': expected, 'first_candle_utc': begin.isoformat(),
            'last_candle_utc': df.time.iloc[-1].isoformat(), 'interval': '15m'}


def prepare(start='2024-01', end='2024-01', out_dir='data'):
    path = Path(download_binance('BTCUSDT', '15m', start, end, out_dir=out_dir))
    df = load_candles(path)
    check = validate_period(df, start, end)
    manifest = {
        'instrument': 'BTCUSDT', 'venue': 'Binance spot', 'timeframe': '15m',
        'market_data_type': 'OHLCV', 'source': 'https://data.binance.vision',
        'months_inclusive': [start, end], 'file_name': path.name,
        'sha256': sha256(path), 'rows': int(len(df)),
        **check,
        'note': 'Public historical market data; not a live trade or performance claim.'
    }
    target = path.with_suffix('.manifest.json')
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Data verified: {len(df):,} candles, SHA-256 {manifest["sha256"]}', flush=True)
    print(f'Manifest: {target}', flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start', default='2024-01')
    ap.add_argument('--end', default='2024-01')
    ap.add_argument('--out-dir', default='data')
    a = ap.parse_args()
    prepare(a.start, a.end, a.out_dir)


if __name__ == '__main__':
    main()
