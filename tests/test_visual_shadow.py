"""Phase 8 training chronology and five-archive integrity regression tests."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data import synthetic
from scripts import train_2024_visual_shadow as T
from scripts import freeze_visual_ensemble as F


def test_fixed_2024_split_and_causal_6_frames():
    # Synthetic prices only test chronology; they cannot support performance claims.
    df=synthetic(35136,kind='null',seed=19)
    panels,labels,tr,va,indices=T.checked_split(df)
    assert panels.shape[1:] == (3,32,4)
    assert tr[0] >= 5
    assert len(labels)==len(indices)
    assert len(tr)>1000 and len(va)>1000
    assert indices[tr[-1]] + 4 < indices[va[0]]
    assert df.time.iloc[indices[tr[-1]]+4] < pd.Timestamp('2024-10-01',tz='UTC')
    assert df.time.iloc[indices[va[0]]] >= pd.Timestamp('2024-10-01',tz='UTC')


def test_full_calendar_year_required():
    df=synthetic(1800,kind='null',seed=11)
    with pytest.raises(ValueError,match='exactly 2024|2024 candles'):
        T.checked_split(df)


def fake_archive(tmp_path):
    for seed in T.SEEDS:
        root=tmp_path/f'visual-seed-{seed}'/f'visual_seed_{seed}'
        root.mkdir(parents=True,exist_ok=True)
        weights=root/f'wm_seed{seed}.pt'
        weights.write_bytes(f'fake-2024-model-weight-seed-{seed}'.encode())
        config={'seed':seed,'K':T.K,'window':T.WINDOW,'img':T.IMAGE_SIZE,
                'epochs_max':T.EPOCHS,'batch_size':T.BATCH_SIZE,'lr':T.LR,
                'lam':T.LAMBDA,'architecture':'WorldModel','mode':'wm','encoder':'cnn'}
        meta={'seed':seed,'data_rows':35136,'dataset_sha256':T.EXPECTED_DATA_SHA256,
              'train_samples':25000,'validation_samples':8500,
              'validation_bce':.690+seed*.0001,
              'checkpoint_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
              'config':config}
        (root/'visual_training.json').write_text(json.dumps(meta))


def test_freeze_requires_every_seed_and_matching_hash(tmp_path):
    fake_archive(tmp_path)
    manifest=F.freeze(tmp_path,tmp_path/'out')
    assert manifest['seeds']==[0,1,2,3,4]
    assert len(manifest['weights_sha256_by_seed'])==5
    import zipfile
    with zipfile.ZipFile(tmp_path/'out'/'frozen_2024_visual_ensemble.zip') as z:
        assert set(z.namelist())=={'wm_seed0.pt','wm_seed1.pt','wm_seed2.pt','wm_seed3.pt','wm_seed4.pt','visual_manifest.json'}
    (tmp_path/'visual-seed-2'/'visual_seed_2'/'wm_seed2.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='corrupted'):
        F.freeze(tmp_path,tmp_path/'out2')


def test_archive_fails_closed_if_seed_missing_or_hyperparams_changed(tmp_path):
    fake_archive(tmp_path)
    import shutil
    shutil.rmtree(tmp_path/'visual-seed-3')
    with pytest.raises(ValueError,match='seed 3'):
        F.freeze(tmp_path,tmp_path/'out')
    fake_archive(tmp_path)
    p=tmp_path/'visual-seed-4'/'visual_seed_4'/'visual_training.json'
    d=json.loads(p.read_text());d['config']['epochs_max']=11
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='epochs_max'):
        F.freeze(tmp_path,tmp_path/'out')
