"""Fail-closed checks for the five-seed *aggregate*, not an accuracy claim."""
import copy
import json

import pytest

from scripts.summarize_full_year import MODELS, collect, summarize, markdown


def fake_metrics(seed=0, sha='b' * 64):
    config={'data':'data/BTCUSDT_15m.csv','models':'phase2',
            'seed':seed,'out':f'results_seed_{seed}','folds':4,'epochs':10,
            'window':32,'img':64,'horizon':4,'K':6,'bs':128,
            'bootstrap':1000,'cost':0.0005}
    models={m:{'accuracy':0.50 + seed*.001,'auc':0.51,
               'brier':0.25,'sharpe_net':-1.0,'psr':0.2,'dsr':0.01}
            for m in MODELS}
    return {'config':config,'models':models,'folds':[{}, {}, {}, {}],
            'n_test':17000,'data_provenance':{
                'sha256':sha,'rows':35136,'synthetic':False,
                'start_utc':'2024-01-01T00:00:00+00:00',
                'end_utc':'2024-12-31T23:45:00+00:00'}}


def write(root, seed, data=None):
    path=root/f'btcusdt-2024-seed-{seed}'/f'results_seed_{seed}'/'metrics.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data or fake_metrics(seed)), encoding='utf-8')


def test_aggregate_five_matching_seeds(tmp_path):
    for s in range(5):write(tmp_path,s)
    result=summarize(collect(tmp_path))
    assert result['status']=='completed'
    assert result['seeds']==[0,1,2,3,4]
    assert result['model_count']==8
    assert 'fusion' in result['models']
    assert 'The model ranking' in markdown(result)


def test_missing_seed_rejected(tmp_path):
    for s in range(4):write(tmp_path,s)
    with pytest.raises(ValueError,match='Missing completed seed'):collect(tmp_path)


def test_mismatched_history_rejected(tmp_path):
    for s in range(5):write(tmp_path,s, fake_metrics(s,'a'*64 if s==3 else 'b'*64))
    with pytest.raises(ValueError,match='Different input CSV'):collect(tmp_path)


def test_duplicate_seed_rejected(tmp_path):
    for s in range(5):write(tmp_path,s)
    extra=tmp_path/'duplicate'/'metrics.json'
    extra.parent.mkdir();extra.write_text(json.dumps(fake_metrics(2)))
    with pytest.raises(ValueError,match='Duplicate'):collect(tmp_path)


def test_config_shift_rejected(tmp_path):
    for s in range(5):write(tmp_path,s, fake_metrics(s))
    data=fake_metrics(4);data['config']['window']=16
    write(tmp_path,4,data)
    with pytest.raises(ValueError,match='expected window=32'):collect(tmp_path)
