from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
from scripts.run_2025_holdout import ensure_locked_range,split_locked

def make_df():
    times=pd.date_range('2024-01-01T00:00:00Z','2025-03-31T23:45:00Z',freq='15min')
    return pd.DataFrame({'time':times})

def test_full_2024_2025q1_range():
    d=make_df();ensure_locked_range(d)
    assert len(d)>40000

def test_rejects_missing_end():
    d=make_df().iloc[:-1]
    with pytest.raises(ValueError):ensure_locked_range(d)

def test_split_does_not_train_on_2025_or_overlap_labels():
    d=make_df();idx=np.arange(511,len(d))
    tr,va,te=split_locked(d,idx,h=4,embargo=10)
    assert idx[tr[-1]]+4<idx[va[0]]
    assert idx[va[-1]]+4<idx[te[0]]
    assert d.time.iloc[idx[tr]+4].max()<pd.Timestamp('2025-01-01T00:00:00Z')
    assert d.time.iloc[idx[va]+4].max()<pd.Timestamp('2025-01-01T00:00:00Z')
    assert d.time.iloc[idx[te]].min()>=pd.Timestamp('2025-01-01T00:00:00Z')

def test_bad_embargo():
    d=make_df()
    with pytest.raises(ValueError):split_locked(d,np.arange(511,len(d)),h=4,embargo=1)


def test_holdout_run_uses_valid_panel_interface(monkeypatch):
    """Catch mismatched panels() keyword names before cloud training begins."""
    import scripts.run_2025_holdout as h
    d = make_df()
    d['open'] = 100.0
    d['high'] = 100.0
    d['low'] = 100.0
    d['close'] = 100.0
    monkeypatch.setattr(h, 'load_candles', lambda path: d)

    class ExpectedStop(Exception):
        pass

    def verify_panel_call(frame, window=32):
        assert frame is d
        assert window == 32
        raise ExpectedStop()

    monkeypatch.setattr(h, 'panels', verify_panel_call)
    with pytest.raises(ExpectedStop):
        h.run('unused.csv', epochs=1, batch_size=4)
