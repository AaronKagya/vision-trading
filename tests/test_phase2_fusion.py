"""Phase 2 model experiments: strictly historical inputs and honest diagnostics."""
import json
import numpy as np
import pytest
import torch
from src.models import FusionModel, fit_predict_fusion
from src.phase2_eval import _comparison, _confidence_bins, _historical_regimes
from src.data import synthetic
from src.features import first_valid_index, panels


def test_fusion_model_rejects_mismatched_modalities():
    m=FusionModel().eval()
    with pytest.raises(ValueError):
        m(torch.randn(2,16,12), torch.randn(3,3,32,32))
    with pytest.raises(ValueError):
        m(torch.randn(2,16,12), torch.randn(2,4,32,32))


def test_fusion_does_not_inspect_next_image_for_a_decision():
    rng=np.random.default_rng(42)
    im=rng.integers(0,256,(12,3,32,32),dtype=np.uint8)
    num=rng.standard_normal((12,16,12)).astype('float32')
    m=FusionModel().eval()
    def forecast(images):
        with torch.no_grad():
            return m(torch.from_numpy(num[7:8]),
                     torch.from_numpy(images[7:8].astype('float32')/255-.5)).item()
    before=forecast(im)
    perturbed=im.copy()
    perturbed[8:]=0
    assert forecast(perturbed)==pytest.approx(before,abs=1e-8)


def test_fusion_fit_uses_training_only_scaler(tmp_path):
    rng=np.random.default_rng(5)
    img=rng.integers(0,256,(55,3,32,32),dtype=np.uint8)
    num=rng.normal(0,.2,(55,16,12)).astype('float32')
    num[42:] += 400  # huge test-only distribution shift must not affect scaler
    y=np.array([0,1]*28)[:55]
    tr=np.arange(0,26); va=np.arange(26,39); te=np.arange(42,55)
    dest=tmp_path/'fusion.pt'
    p=fit_predict_fusion(img,num,y,tr,va,te,epochs=1,lr=.001,bs=8,seed=3,checkpoint=dest)
    assert p.shape==(len(te),) and np.isfinite(p).all()
    assert np.all((p>=0)&(p<=1))
    ck=torch.load(dest,map_location='cpu',weights_only=True)
    np.testing.assert_allclose(ck['scaler_mean'],num[tr].reshape(-1,12).mean(0),rtol=1e-5,atol=1e-7)
    assert ck['architecture']=='causal_fusion_v1'


def test_phase2_comparison_and_fixed_confidence_intervals():
    p=np.array([.95,.5,.51,.61,.20,.55])
    q=np.array([.2,.8,.6,.3,.6,.8])
    y=np.array([1,0,1,1,0,1])
    rows=_confidence_bins(p,y)
    assert sum(x['observations'] for x in rows)==len(y)
    d=_comparison(np.tile(p,50),np.tile(q,50),np.tile(y,50),bootstraps=20)
    assert len(d['ci95_points'])==2
    assert d['method'].startswith('paired')


def test_volatility_regimes_calibrate_only_on_pretest_prices():
    df=synthetic(850,seed=1)
    first=first_valid_index(16)
    # Two contiguous historical test blocks and training indices preceding each.
    idx=np.arange(670,770)
    rng=np.random.default_rng(1)
    pred={'fusion':rng.random(len(idx))}
    yy=rng.integers(0,2,len(idx))
    folds=[{'fold':0,'test_n':50,'train_last_sample':70},
           {'fold':1,'test_n':50,'train_last_sample':140}]
    report={'config':{'window':16}}
    base=_historical_regimes(report,idx,df.close.to_numpy(),folds,pred,yy)
    revised=df.close.to_numpy().copy()
    revised[790:]*=10  # future-only changes must not affect past terciles
    again=_historical_regimes(report,idx,revised,folds,pred,yy)
    assert base==again
    assert len(base)==2
