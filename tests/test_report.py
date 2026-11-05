"""Ensure the HTML dashboard generates without accidental string interpolation."""
import json
from src.report import build


def test_report_renders_valid_html(tmp_path):
    (tmp_path/'metrics.json').write_text(json.dumps({
        'n_test':20,'up_rate_test':.5,'n_trials':7,
        'config':{'folds':2,'data':'synthetic_null','seed':1,'horizon':4,'encoder':'cnn'},
        'data_provenance':{'synthetic':True,'sha256':None},
        'models':{'wm':{'accuracy':.5,'acc_ci':[.4,.6],'auc':.52,'sharpe_net':-1.2,'dsr':.02}}}))
    path=build(tmp_path)
    content=path.read_text()
    assert 'Research-only' in content and '50.00%' in content
    assert 'Synthetic simulation' in content
    assert '{n_test' not in content
