"""Test causal splits, actual epoch logging, fail-closed selection and read-only Q2 inference."""
import copy, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from src.data import synthetic
from src.epoch_audit import (BUDGETS,make_audit_data,fit_audit,save_checkpoints,
                             predict_checkpoint,LSTMBaseline)
from scripts.run_epoch_audit import run
from scripts.summarize_epoch_audit import collect
from scripts.score_epoch_holdout import first_2024_sha

@pytest.fixture(scope='module')
def year():
    # Synthetic candles are used exclusively for software unit testing.
    return synthetic(n=35136,kind='null',seed=15)

def test_year_end_and_train_val_leakage(year):
    num,y,tr,va,idx=make_audit_data(year)
    assert len(num)==len(idx)
    assert idx[tr[-1]]+4 < idx[va[0]]
    assert year.time.iloc[idx[tr]+4].max()<pd.Timestamp('2024-10-01T00:00:00Z')
    assert year.time.iloc[idx[va]].min()>=pd.Timestamp('2024-10-01T00:00:00Z')
    assert year.time.iloc[idx[va]+4].max()<pd.Timestamp('2025-01-01T00:00:00Z')

def test_missing_year_or_candle_rejected(year):
    with pytest.raises(ValueError): make_audit_data(year.iloc[:-1])
    with pytest.raises(ValueError): make_audit_data(year.drop(index=132).reset_index(drop=True))

def test_history_tracks_actual_epochs_and_best_validation_snapshot(tmp_path):
    rng=np.random.default_rng(4)
    num=rng.normal(0,.3,size=(190,8,12)).astype('float32')
    labels=(rng.random(190)>.5).astype(int)
    tr=np.arange(0,115);va=np.arange(145,175)
    histories,shots,scaler=fit_audit(num,labels,tr,va,seed=2,budgets=(1,2,4),max_epochs=4,
                                    bs=32,patience=5)
    assert len(histories)==4
    assert [x['epoch'] for x in histories]==[1,2,3,4]
    for budget in (1,2,4):
        assert 1 <= shots[budget]['selected_epoch'] <= budget
        assert np.isclose(shots[budget]['best_validation_bce'],
                          histories[shots[budget]['selected_epoch']-1]['validation_bce'])
    assert np.isfinite(scaler).all()

def test_checkpoint_predict_and_hash_fence(tmp_path):
    model=LSTMBaseline()
    sd=np.ones(12,dtype=float)
    cfg={'seed':0,'budgets':list(BUDGETS),'max_epochs':40}
    shots={5:{'state_dict':copy.deepcopy(model.state_dict()),'selected_epoch':1,'best_validation_bce':.69}}
    save_checkpoints(tmp_path,shots,sd,config=cfg,data_sha256='hash123')
    p=tmp_path/'checkpoint_budget_5.pt'
    arr=np.ones((12,32,12),dtype=np.float32)
    predictions=predict_checkpoint(p,arr,np.array([2,7]),expected_budget=5,expected_data_sha='hash123')
    assert len(predictions)==2 and np.isfinite(predictions).all()
    with pytest.raises(ValueError):predict_checkpoint(p,arr,np.array([2]),expected_budget=5,expected_data_sha='different')
    with pytest.raises(ValueError):predict_checkpoint(p,arr,np.array([2]),expected_budget=10,expected_data_sha='hash123')

def create_fixture_audit(root, seed, losses, datahash='hash123', cfg_override=None):
    path=root/f'epoch-seed-{seed}';path.mkdir(parents=True)
    config={'seed':seed,'budgets':list(BUDGETS),'max_epochs':40,'window':32,'lr':.001,
            'train_last_utc':'2024-09-30','validation_first_utc':'2024-10-01'}
    if cfg_override:config.update(cfg_override)
    h=[{'epoch':i,'validation_bce':.8-i*.01} for i in range(1,11)]
    budgets={str(k):{'selected_epoch':min(k,10),'validation_bce':h[min(k,10)-1]['validation_bce']}
             for k in BUDGETS}
    if losses:
        for b,v in losses.items():
            # Set an explicit loss at its selected epoch; cap-20 and cap-40
            # share epoch-10 and thus must agree.
            budgets[str(b)]['validation_bce']=v
    doc={'seed':seed,'data_sha256_2024':datahash,'data_rows':35136,'n_train':24000,
         'n_validation':8500,'config':config,'history':h,'epochs_executed':len(h),'budgets':budgets}
    (path/'audit.json').write_text(json.dumps(doc))
    for b in BUDGETS:(path/f'checkpoint_budget_{b}.pt').write_text('unit-test sentinel')

def test_aggregate_rejects_missing_seed_or_mixed_hash(tmp_path):
    for seed in range(4):create_fixture_audit(tmp_path,seed,{})
    with pytest.raises(ValueError,match='Missing seed'):collect(tmp_path)
    create_fixture_audit(tmp_path,4,{},datahash='DIFFERENT')
    with pytest.raises(ValueError,match='Mixed data provenance'):collect(tmp_path)

def test_aggregate_selects_min_validation_not_test(tmp_path):
    for seed in range(5):create_fixture_audit(tmp_path,seed,{})
    out=collect(tmp_path)
    assert out['selected_max_epochs']==10 # 10,20,40 tie: prefer shorter budget
    assert out['data_rows_2024']==35136
    assert out['seeds']==[0,1,2,3,4]

def test_2024_raw_hash_prefix(tmp_path):
    data=b'time,open\na,b\nc,d\n2025,z\n'
    p=tmp_path/'candles.csv';p.write_bytes(data)
    assert first_2024_sha(p,n=2)==hashlib.sha256(b'time,open\na,b\nc,d\n').hexdigest()
    with pytest.raises(ValueError):first_2024_sha(p,n=8)
