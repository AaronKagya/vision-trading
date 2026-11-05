import numpy as np
import pytest
from src.abstention import positions_from_probabilities,evaluate_policy

def test_confident_up_down_and_flat():
    assert positions_from_probabilities([.7,.3,.5,.54,.46],.05).tolist()==[1,-1,0,0,0]

def test_long_only_never_shorts():
    assert positions_from_probabilities([.7,.3,.5],.05,long_only=True).tolist()==[1,0,0]

def test_everything_flat_has_zero_log_profit():
    x=evaluate_policy([.5,.51,.49,.5],[1,0,1,0],[.01,-.01,.01,.01],horizon=1)
    assert x['coverage']==0 and x['net_log_pnl_proxy']==0 and x['selected_direction_accuracy'] is None

def test_terminal_flatten_charges_cost():
    x=evaluate_policy([.8,.8],[1,1],[.01,.02],horizon=1,cost=.001)
    assert x['turnover_units_including_flatten']==2
    assert x['net_log_pnl_proxy']==pytest.approx(.028)

def test_stride_nonoverlap():
    x=evaluate_policy([.8,.8,.8,.8],[1,1,1,1],[.02,.08,.03,.1],horizon=2,cost=0)
    assert x['nonoverlap_intervals']==2
    assert x['net_log_pnl_proxy']==pytest.approx(.05)

@pytest.mark.parametrize('p,d', [([1.3],.05),([float('nan')],.05),([.4],-.1),([.4],.6)])
def test_invalid_confidence_input(p,d):
    with pytest.raises(ValueError):positions_from_probabilities(p,d)

def test_invalid_returns():
    with pytest.raises(ValueError):evaluate_policy([.8],[1],[float('nan')])
