import io, json, zipfile
import numpy as np
import pytest
from src.abstention import evaluate_policy
from src.paper_replay import replay,ensure_matches_archived_policy
from scripts.build_paper_replay import build_payload


def test_flat_predictions_pay_nothing():
    r=replay([.5]*8,[.02]*8)
    assert r['summary']['n_intervals']==2
    assert r['summary']['n_flats']==2
    assert r['summary']['net_log_pnl_proxy']==0.


def test_terminal_flatten_and_short_reversal():
    v=replay([.6,.5,.5,.5,.4,.5,.5,.5],[.01]*8,horizon=4)
    assert list(v['positions'])==[1,-1]
    assert v['summary']['turnover']==4.0  # enter +1, reverse +1 -> -1, terminal exit
    assert v['summary']['net_log_pnl_proxy']==pytest.approx(-.002)


def test_nonoverlap_no_double_count():
    p=np.array([.8]*12);r=np.array([.01]*12);a=replay(p,r,horizon=4)
    assert a['summary']['n_intervals']==3
    assert a['summary']['gross_log_pnl_proxy']==pytest.approx(.03)
    assert a['summary']['turnover']==2
    assert a['summary']['net_log_pnl_proxy']==pytest.approx(.029)


def test_bad_inputs_fail_closed():
    for p,r in [([.5],[float('nan')]),([1.2],[.01]),([],[]),([[.5]],[.01]),([.6,.7],[.02])]:
        with pytest.raises(ValueError):replay(p,r)


def test_matches_research_policy():
    p=np.array([.6,.4,.5,.7]*6);r=np.linspace(-.03,.02,len(p));y=(r>0).astype(int)
    official=evaluate_policy(p,y,r,horizon=4,distance=.05,cost=.0005)
    replica=ensure_matches_archived_policy(p,y,r,official)
    assert replica['summary']['net_log_pnl_proxy']==pytest.approx(official['net_log_pnl_proxy'])
    altered=dict(official,net_log_pnl_proxy=0.3)
    with pytest.raises(ValueError,match='disagree'):ensure_matches_archived_policy(p,y,r,altered)


def test_q1_q2_provenance_real_archives_if_available():
    # CI tests do not depend on unreleased / expiring GitHub artifacts.
    from pathlib import Path
    folder=Path(__file__).resolve().parents[3]
    q1=folder/'Vision_Trading_2025Q1_Holdout_Results.zip'
    selection=folder/'Vision_Trading_Phase4_selection.zip'
    q2=folder/'Vision_Trading_Phase4_holdout.zip'
    if not all(x.exists() for x in (q1,selection,q2)):
        pytest.skip('Official artifact ZIPs must be provided explicitly for integration test')
    payload=build_payload(q1,selection,q2)
    assert len(payload['rows'])==2159
    assert payload['summary']['net_log_pnl_proxy']==pytest.approx(-.30326038812017153)
    assert payload['q2']['selected_max_epochs']==20
