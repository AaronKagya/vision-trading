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
