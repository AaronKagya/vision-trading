"""No network, no trading, no look-ahead; tests for prospective AI shadow monitor."""
import copy
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src import forward_ai as A


def history(last_close_ms, n=700, up=True):
    stamps=np.arange(last_close_ms-n*A.INTERVAL_MS,last_close_ms,A.INTERVAL_MS,dtype=np.int64)
    close=110+np.arange(n)*.03*(1 if up else -1)
    op=close-.02
    frame=pd.DataFrame({'open_ms':stamps,'open':op,'high':np.maximum(op,close)+.09,
                        'low':np.minimum(op,close)-.09,'close':close,
                        'volume':np.ones(n)*100})
    frame['time']=pd.to_datetime(frame.open_ms,unit='ms',utc=True)
    return frame


def milliseconds(s):
    return int(pd.Timestamp(s).timestamp()*1000)


MANIFEST={'model_id':A.MODEL,'weights_sha256_by_seed':{'0':'a'},'training_data_sha256':A.EXPECTED_DATA_HASH}


def fake_predict(df):
    return .6,[.5,.55,.6,.65,.7]


def test_no_forecast_at_activation():
    t=milliseconds('2026-11-14T10:00:00Z')
    s=A.advance(None,history(t),t+7*60000,MANIFEST,fake_predict)
    assert s['forecasts']==[]
    assert s['activated_at_ms']==t+7*60000
    assert A.public_summary(s)['metrics']['settled']==0


def test_forward_prediction_only_on_new_fresh_hour():
    t=milliseconds('2026-11-14T10:00:00Z')
    s=A.advance(None,history(t),t+7*60000,MANIFEST)
    v=A.advance(s,history(t+A.HORIZON_MS),t+A.HORIZON_MS+7*60000,MANIFEST,fake_predict)
    assert len(v['forecasts'])==1
    e=v['forecasts'][0]
    assert e['origin_close_ms']==t+A.HORIZON_MS
    assert e['target_close_ms']==t+2*A.HORIZON_MS
    assert e['status']=='pending'
    assert e['prob_up']==.6 and e['signal_type']=='model_observation_not_trade'
    assert s['forecasts']==[]
    assert 'trade' not in e


def test_same_hour_idempotent_and_late_poll_no_backfill():
    t=milliseconds('2026-11-14T10:00:00Z')
    s=A.advance(None,history(t),t+7*60000,MANIFEST)
    u=A.advance(s,history(t+A.HORIZON_MS),t+A.HORIZON_MS+7*60000,MANIFEST,fake_predict)
    v=A.advance(u,history(t+A.HORIZON_MS),t+A.HORIZON_MS+9*60000,MANIFEST,fake_predict)
    assert len(v['forecasts'])==1
    stale=A.advance(s,history(t+A.HORIZON_MS),t+A.HORIZON_MS+14*60000,MANIFEST,fake_predict)
    assert stale['forecasts']==[]


def test_settle_only_after_future_candle_closed():
    t=milliseconds('2026-11-14T10:00:00Z')
    s=A.advance(None,history(t),t+7*60000,MANIFEST)
    s=A.advance(s,history(t+A.HORIZON_MS),t+A.HORIZON_MS+7*60000,MANIFEST,fake_predict)
    assert s['forecasts'][0]['status']=='pending'
    interim=A.advance(s,history(t+A.HORIZON_MS+3*A.INTERVAL_MS),t+A.HORIZON_MS+3*A.INTERVAL_MS+7*60000,MANIFEST)
    assert interim['forecasts'][0]['status']=='pending'
    final=A.advance(interim,history(t+2*A.HORIZON_MS),t+2*A.HORIZON_MS+7*60000,MANIFEST,fake_predict)
    assert final['forecasts'][0]['status']=='settled'
    assert final['forecasts'][0]['label_observed_at_ms']==t+2*A.HORIZON_MS+7*60000
    m=A.public_summary(final)['metrics']
    assert m['settled']==1 and m['accuracy'] in (0.0,1.0)
    assert m['momentum_baseline_accuracy'] in (0.0,1.0)


def test_fail_closed_on_model_swap_or_time_regression():
    t=milliseconds('2026-11-14T10:00:00Z')
    s=A.advance(None,history(t),t+7*60000,MANIFEST)
    swapped={**MANIFEST,'model_id':'other'}
    with pytest.raises(ValueError,match='Model version mismatch'):
        A.advance(s,history(t),t+8*60000,swapped)
    with pytest.raises(ValueError,match='time regression'):
        A.advance(s,history(t),t+6*60000,MANIFEST)


def test_fail_closed_on_bad_forecasts():
    t=milliseconds('2026-11-14T10:00:00Z')
    s=A.advance(None,history(t),t+7*60000,MANIFEST)
    with pytest.raises(ValueError,match='Ensemble value differs'):
        A.advance(s,history(t+A.HORIZON_MS),t+A.HORIZON_MS+7*60000,MANIFEST,
                  lambda df:(.9,[.5]*5))


def test_closed_candles_ignore_unfinished_and_bad_gaps():
    t=milliseconds('2026-11-14T11:00:00Z')
    df=history(t)
    rows=[[int(r.open_ms),str(r.open),str(r.high),str(r.low),str(r.close),str(r.volume)] for r in df.itertuples(index=False)]
    rows.append([t,'2','2','2','2','2'])
    x=A.validated_closed_klines(rows,t+7*60000)
    assert x.iloc[-1].open_ms==t-A.INTERVAL_MS
    assert len(x)==700
    rows.pop(10)
    with pytest.raises(ValueError,match='Missing, repeated'):
        A.validated_closed_klines(rows,t+7*60000)


def test_all_five_frozen_models_are_loadable_and_agree_with_enforced_mean():
    base=Path(__file__).resolve().parents[1]
    root=base/'models/phase4_2024'
    manifest=A.verified_checkpoint_manifest(root)
    assert set(manifest['weights_sha256_by_seed'])==set(map(str,range(5)))
    models=A.verified_models(root)
    t=milliseconds('2026-11-14T11:00:00Z')
    p,individual=A.predict_frozen_ensemble(history(t),models)
    assert len(individual)==5 and math.isfinite(p)
    assert p==pytest.approx(sum(individual)/5,abs=1e-8)
    assert 0<=p<=1
