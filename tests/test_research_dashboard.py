import io
import json
import zipfile
import numpy as np
import pytest
from scripts.build_research_dashboard import make_payload,build


def mock_sources(tmp_path,broken=False):
    holdout=tmp_path/'holdout.zip'
    summary=tmp_path/'summary.zip'
    met={'n_test':8,'training_must_not_use_2025':True,
         'external_test_period':'2025-01-01 through 2025-03-31, 15-minute candles',
         'models':{k:{'accuracy':0.5} for k in ('lstm_ohlc','logreg_lags','majority')}}
    data=io.BytesIO();d=dict(y=np.array([0,1]*4),fwd=np.zeros(8),t_idx=np.arange(100,108))
    for name in ('lstm_ohlc','logreg_lags','majority'):
        d['p_'+name]=np.full(8,0.52)
    if broken:d['p_lstm_ohlc']=np.full(8,1.5)
    np.savez_compressed(data,**d)
    with zipfile.ZipFile(holdout,'w') as z:
        z.writestr('results/metrics.json',json.dumps(met))
        z.writestr('results/preds.npz',data.getvalue())
    sm={'status':'completed','seeds':[0,1,2,3,4],'model_count':8,
        'models':{k:{} for k in ('lstm_ohlc','fusion','wm','majority')}}
    with zipfile.ZipFile(summary,'w') as z:z.writestr('summary.json',json.dumps(sm))
    return holdout,summary


def test_dashboard_payload_from_completed_artifacts(tmp_path):
    h,s=mock_sources(tmp_path)
    p=make_payload(h,s)
    assert p['n_test']==8
    assert len(p['predictions']['lstm_ohlc'])==8
    assert p['benchmark']['model_count']==8
    assert len(p['sources']['holdout_sha256'])==64


def test_dashboard_rejects_invalid_probability(tmp_path):
    h,s=mock_sources(tmp_path,broken=True)
    with pytest.raises(ValueError,match='Invalid probabilities'):
        make_payload(h,s)


def test_dashboard_renders_offline_html(tmp_path):
    h,s=mock_sources(tmp_path)
    dest=tmp_path/'index.html'
    build(h,s,dest)
    html=dest.read_text()
    assert '@@EVIDENCE_JSON@@' not in html
    assert 'Research-only demonstration' in html
    assert 'id="evidence"' in html
    assert 'BTCUSDT' in html
