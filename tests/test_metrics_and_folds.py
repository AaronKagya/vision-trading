"""Guard time splits, terminology, and research-only proxy calculations."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.metrics import strategy_returns, compounded_from_log_proxy, block_bootstrap
from src.run_experiments import build_folds, calculate_metrics


def test_walk_forward_embargoes_and_disjoint_chronological_tests():
    samples = np.arange(5, 2200)
    ff = list(build_folds(samples, folds=4, horizon=4, K=6))
    all_tests = np.concatenate([f['te'] for f in ff])
    assert len(all_tests) == len(np.unique(all_tests))
    for f in ff:
        assert f['tr'].max()+4 < f['va'].min()
        assert f['va'].max()+4 < f['te'].min()
        assert all(np.diff(f['tr']) == 1)


def test_transaction_costs_account_for_position_flips():
    up_probs = np.array([0.8, 0.2, 0.2, 0.9])
    fwd = np.array([.01, -.01, -.01, .01])
    out = strategy_returns(up_probs, fwd, cost=.001, stride=1)
    np.testing.assert_allclose(out, [.009, .008, .01, .008])
    assert compounded_from_log_proxy(.482165) == pytest.approx(np.expm1(.482165))


def test_metrics_do_not_confuse_log_pnl_proxy_with_equity_return():
    p={'lstm_ohlc':np.array([.8,.7,.4,.9]*40),
       'wm':np.array([.6,.2,.8,.4]*40)}
    y=np.tile([1,1,0,1], 40)
    fwd=np.tile([.003,-.001,.004,.002], 40)
    cfg={'horizon':4,'cost':.0005,'extra_trials':0}
    res=calculate_metrics(p,y,fwd,cfg,bootstrap=20)
    assert res['models']['wm']['total_return'] == res['models']['wm']['total_log_pnl_proxy']
    assert res['models']['wm']['implied_compounded_return'] == pytest.approx(
        np.expm1(res['models']['wm']['total_log_pnl_proxy']))
    assert 'NOT simple equity' in res['metric_notes']['total_return']
    assert res['models']['wm']['n_trades'] == 40


def test_block_bootstrap_supports_small_series():
    ci=block_bootstrap(lambda idx: float(idx.mean()),n=12,block=50,B=10)
    assert len(ci)==2 and ci[0]<=ci[1]
