"""Detect invalid input data before an experiment reaches a trained model."""
import pandas as pd
import pytest
from src.data import synthetic, validate_candles


def test_synthetic_candles_are_valid_and_utc():
    data = synthetic(530, seed=5)
    assert len(data) == 530
    assert str(data.time.dt.tz) == 'UTC'


@pytest.mark.parametrize('corruption,match', [
    ('duplicate','Duplicate'),
    ('missing_interval','Missing/irregular'),
    ('invalid_high','Inconsistent'),
    ('negative_price','Invalid nonpositive'),
    ('missing_close','Nonfinite'),
])
def test_bad_candles_rejected(corruption, match):
    df = synthetic(530, seed=10)
    if corruption == 'duplicate':
        df.loc[20, 'time'] = df.loc[19, 'time']
    elif corruption == 'missing_interval':
        df = df.drop(index=20)
    elif corruption == 'invalid_high':
        df.loc[25, 'high'] = df.loc[25, 'low']
    elif corruption == 'negative_price':
        df.loc[25, 'close'] = -1
    elif corruption == 'missing_close':
        df.loc[25, 'close'] = float('nan')
    with pytest.raises(ValueError, match=match):
        validate_candles(df)
