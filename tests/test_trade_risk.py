import pytest
from src.trade_risk import RiskRequest,plan
from src.advisor_signals import CandidateSignal,validate_candidate


def example(**kwargs):
    args=dict(balance=500,risk_percent=1,entry=1000,stop=990,take=1020,side='buy',
              account_value_per_price_point_per_lot=1,min_lots=0.01,lot_step=0.01,max_lots=5,
              roundtrip_fees_per_lot=0,spread_and_slippage_points=0)
    args.update(kwargs)
    return RiskRequest(**args)


def test_size_and_rr():
    x=plan(example())
    assert x['volume_lots']==pytest.approx(.5)
    assert x['max_loss']==pytest.approx(5)
    assert x['risk_reward_net']==pytest.approx(2)
    assert x['execution_enabled'] is False


def test_costs_reduce_size_not_hide_loss():
    x=plan(example(roundtrip_fees_per_lot=2,spread_and_slippage_points=1))
    assert x['volume_lots']==pytest.approx(.38)
    assert x['max_loss']<=5
    assert x['risk_reward_net']<2


def test_minimum_lot_risk_cap_blocks():
    r=plan(example(balance=10,risk_percent=.5,min_lots=.1))
    assert r['status']=='no_trade'
    assert r['volume_lots']==0


def test_wrong_direction_and_missing_spec_rejected():
    with pytest.raises(ValueError,match='Buy plans'):
        plan(example(stop=1100))
    with pytest.raises(ValueError,match='finite'):
        plan(example(account_value_per_price_point_per_lot=0))
    with pytest.raises(ValueError,match='2%'):
        plan(example(risk_percent=10))


def test_sell_side_and_step_rounding():
    r=plan(example(side='sell',stop=1010,take=980,lot_step=.2,min_lots=.1))
    assert r['volume_lots']==pytest.approx(.5)


def test_signal_provider_requires_provenance_and_has_no_execution():
    s=CandidateSignal('Provider','Jump 75 Index','model-1',1000,3600,'up',.54,995)
    x=validate_candidate(s,observed_epoch=1030)
    assert x['allow_orders'] is False
    with pytest.raises(ValueError,match='Future data'):
        validate_candidate(CandidateSignal('Provider','R','x',1000,3600,'up',.54,1001),observed_epoch=1030)
    with pytest.raises(ValueError,match='Stale'):
        validate_candidate(s,observed_epoch=3000)
