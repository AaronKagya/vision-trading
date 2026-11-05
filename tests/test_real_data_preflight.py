"""Offline validation of historical market data range requirements."""
import pandas as pd
import pytest
from scripts.prepare_btcusdt import validate_period


def _candles():
    ts = pd.date_range('2024-01-01', '2024-02-01', inclusive='left', freq='15min', tz='UTC')
    return pd.DataFrame({'time': ts})


def test_complete_month_passes():
    r = validate_period(_candles(), '2024-01', '2024-01')
    assert r['candles'] == 2976


def test_missing_middle_bar_rejected():
    df = _candles().drop(index=100).reset_index(drop=True)
    with pytest.raises(ValueError, match='Expected'):
        validate_period(df, '2024-01', '2024-01')


def test_missing_first_bar_rejected():
    with pytest.raises(ValueError):
        validate_period(_candles().iloc[1:].reset_index(drop=True), '2024-01', '2024-01')


def test_extra_bar_rejected():
    df = _candles()
    extra = pd.DataFrame({'time': [pd.Timestamp('2024-02-01', tz='UTC')]})
    with pytest.raises(ValueError):
        validate_period(pd.concat([df, extra], ignore_index=True), '2024-01', '2024-01')


def test_reversed_dates_rejected():
    with pytest.raises(ValueError, match='End month'):
        validate_period(_candles(), '2024-02', '2024-01')
