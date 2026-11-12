import copy
import json
import pytest
from pathlib import Path
from scripts.aggregate_multiasset import aggregate, validate, EXPECTED_MODELS
from scripts.prepare_multiasset import prepare, ALLOWED

BASE = Path(__file__).resolve().parents[1] / 'evidence'
BTC0 = BASE / 'btcusdt_2024_seed0_metrics.json'
BTC5 = BASE / 'btcusdt_2024_five_seed_summary.json'


def synth_artifacts(tmp_path, seeds=(0,)):
    original=json.loads(BTC0.read_text())
    for sym in ('ETHUSDT','SOLUSDT'):
        for seed in seeds:
            item=copy.deepcopy(original)
            item['config']['seed']=seed
            item['config']['data']=f'data/{sym}_15m.csv'
            item['data_provenance']['sha256']=('b' if sym=='ETHUSDT' else 'c')*64
            path=tmp_path/f'multiasset-{sym}-seed-{seed}'/'results'/'metrics.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(item))
    return tmp_path


def test_initial_dashboard_is_honest(tmp_path):
    d=aggregate(tmp_path,BTC0,BTC5,initial=True)
    assert d['status']=='baseline_only'
    assert d['markets']['ETHUSDT']['status']=='pending'
    assert d['markets']['SOLUSDT']['status']=='pending'
    assert 'accuracy_mean' not in d['markets']['ETHUSDT']
    assert len(d['markets']['BTCUSDT']['models']) == 8


def test_pilot_same_settings_and_markets(tmp_path):
    root=synth_artifacts(tmp_path)
    d=aggregate(root,BTC0,BTC5)
    assert d['status']=='pilot'
    assert set(d['markets']) == {'BTCUSDT','ETHUSDT','SOLUSDT','XAUUSD','EURUSD','VOLATILITY75'}
    assert d['markets']['ETHUSDT']['models']['lstm_ohlc']['accuracy_mean'] > 0
    assert d['markets']['SOLUSDT']['seeds']==[0]
    assert d['markets']['SOLUSDT']['models']['lstm_ohlc']['accuracy_sd'] is None


def test_five_seeds_and_full_btc_reference(tmp_path):
    root=synth_artifacts(tmp_path,tuple(range(5)))
    d=aggregate(root,BTC0,BTC5,tuple(range(5)))
    assert d['status']=='five_seed'
    assert d['markets']['BTCUSDT']['seeds']==[0,1,2,3,4]
    assert all(d['markets'][sym]['seeds']==[0,1,2,3,4] for sym in ('ETHUSDT','SOLUSDT'))


def test_reject_incomplete_results(tmp_path):
    synth_artifacts(tmp_path)
    for f in (tmp_path/'multiasset-SOLUSDT-seed-0').rglob('metrics.json'):
        f.unlink()
    with pytest.raises(ValueError,match='Expected'):
        aggregate(tmp_path,BTC0,BTC5)


def test_reject_modified_configuration(tmp_path):
    synth_artifacts(tmp_path)
    path=next((tmp_path/'multiasset-ETHUSDT-seed-0').rglob('metrics.json'))
    item=json.loads(path.read_text());item['config']['epochs']=1
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError,match='epochs'):
        aggregate(tmp_path,BTC0,BTC5)


def test_reject_wrong_period_or_synthetic(tmp_path):
    item=json.loads(BTC0.read_text());item['data_provenance']['rows']=35000
    with pytest.raises(ValueError,match='coverage'):
        validate(item,0,'BTCUSDT',check_hash=False)
    item=json.loads(BTC0.read_text());item['data_provenance']['synthetic']=True
    with pytest.raises(ValueError,match='fingerprint'):
        validate(item,0,'BTCUSDT',check_hash=False)


def test_reject_different_data_hashes_among_seeds(tmp_path):
    synth_artifacts(tmp_path,tuple(range(5)))
    path=next((tmp_path/'multiasset-ETHUSDT-seed-1').rglob('metrics.json'))
    item=json.loads(path.read_text());item['data_provenance']['sha256']='d'*64
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError,match='hashes'):
        aggregate(tmp_path,BTC0,BTC5,tuple(range(5)))


def test_reject_invalid_or_unapproved_symbols():
    assert 'BTCUSDT' in ALLOWED and 'ETHUSDT' in ALLOWED and 'SOLUSDT' in ALLOWED
    with pytest.raises(ValueError,match='Unapproved'):
        prepare('XAUUSD')


def test_reject_unknown_seeds_and_false_comparisons(tmp_path):
    with pytest.raises(ValueError,match='Invalid seed'):
        aggregate(tmp_path,BTC0,BTC5,(0,0),initial=True)
    with pytest.raises(ValueError,match='Supported comparisons'):
        aggregate(tmp_path,BTC0,BTC5,(0,1),initial=False)
