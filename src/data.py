"""Causal OHLCV data import, validation and illustrative synthetic generators."""
from __future__ import annotations
import io
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
import pandas as pd

COLUMNS = ['time', 'open', 'high', 'low', 'close', 'volume']

def validate_candles(df: pd.DataFrame, *, strict: bool = True, interval: str = '15min') -> pd.DataFrame:
    """Return sorted candles; reject NaNs, duplicates, invalid OHLC and time gaps by default."""
    missing = [c for c in COLUMNS if c not in df]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    out = df[COLUMNS].copy()
    out['time'] = pd.to_datetime(out['time'], utc=True, errors='raise')
    for col in COLUMNS[1:]:
        out[col] = pd.to_numeric(out[col], errors='raise')
    out = out.sort_values('time', kind='stable').reset_index(drop=True)
    if out.time.duplicated().any():
        raise ValueError('Duplicate candle timestamps; correct the source before training.')
    numeric = out[COLUMNS[1:]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError('Nonfinite price or volume values detected.')
    if (out[['open', 'high', 'low', 'close']] <= 0).any().any() or (out.volume < 0).any():
        raise ValueError('Invalid nonpositive price or negative volume.')
    high_max = out[['open', 'close', 'low']].max(axis=1)
    low_min = out[['open', 'close', 'high']].min(axis=1)
    if (out.high + 1e-8 < high_max).any() or (out.low - 1e-8 > low_min).any():
        raise ValueError('Inconsistent OHLC candle(s).')
    if strict and len(out) >= 2:
        expected = pd.Timedelta(interval)
        gaps = out.time.diff().iloc[1:] != expected
        if gaps.any():
            bad_ts = out.time.iloc[int(np.where(gaps)[0][0]) + 1]
            raise ValueError(f'Missing/irregular candles near {bad_ts}; fill or explicitly resolve gaps.')
    if len(out) < 60:
        raise ValueError('Not enough candles for training or feature generation.')
    return out

def load_candles(path: str | Path, *, strict: bool = True) -> pd.DataFrame:
    return validate_candles(pd.read_csv(path), strict=strict)

def download_binance(symbol='BTCUSDT', interval='15m', start='2024-01', end='2024-12', out_dir='data') -> str:
    """Download Binance SPOT monthly klines; fail if a requested month is missing.

    Check exchange API/data licensing terms, regional access rules and data integrity.
    Does not attempt live trades or use exchange credentials.
    """
    frames = []
    for m in pd.period_range(start, end, freq='M'):
        url = f'https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{m}.zip'
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                raw = resp.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                name = next(n for n in z.namelist() if n.endswith('.csv'))
                t = pd.read_csv(z.open(name), header=None).iloc[:, :6]
        except Exception as e:
            raise RuntimeError(f'Download failed for {m}: {url}\n{e}') from e
        t.columns = COLUMNS
        n = pd.to_numeric(t['time'], errors='raise').astype('int64')
        # Recent Binance archives may use microseconds instead of milliseconds.
        t['time'] = pd.to_datetime(n, unit='us' if int(n.iloc[0]) > 10**14 else 'ms', utc=True)
        frames.append(t)
        print(f'Loaded month {m}: {len(t):,} candles')
    df = validate_candles(pd.concat(frames, ignore_index=True), interval={'15m':'15min','1h':'1h'}.get(interval,interval))
    path = Path(out_dir) / f'{symbol}_{interval}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f'Saved {len(df):,} candles to {path}')
    return str(path)

def synthetic(n=8000, kind='null', seed=0):
    """Price simulation for software smoke tests only; never evidence of trading skill."""
    if kind not in ('null', 'planted'):
        raise ValueError('kind must be null or planted')
    rng = np.random.default_rng(seed)
    vol = np.empty(n)
    v = .002
    for i in range(n):
        v = .98*v + .02*.002 + .0003*abs(rng.normal())
        vol[i] = v
    r = np.zeros(n)
    for i in range(n):
        drift = .12*vol[i]*np.sign(r[i-16:i].sum()) if kind=='planted' and i>16 else 0.
        r[i] = drift + vol[i]*rng.normal()
    close = 30000*np.exp(np.cumsum(r))
    opening = np.r_[30000., close[:-1]]
    spread = abs(rng.normal(size=n))*vol*close*.6
    return validate_candles(pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='15min', tz='UTC'),
        'open': opening, 'high': np.maximum(opening,close)+spread,
        'low': np.minimum(opening,close)-spread, 'close': close,
        'volume': rng.uniform(1,10,size=n)}))
