"""Fast end-to-end smoke run of the causal experiment pipeline on synthetic data."""
import json
from pathlib import Path
import subprocess
import sys
import os


def test_synthetic_experiment_runs_and_writes_predictions(tmp_path):
    target=tmp_path/'smoke'
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    cmd=[sys.executable,'-m','src.run_experiments',
         '--data','synthetic_null','--n-synth','680','--window','16','--img','32','--K','2','--folds','1','--epochs','1',
         '--models','majority,persistence,logreg_lags,lstm_ohlc,cnn_static,wm_nolatent,wm',
         '--bs','16','--bootstrap','20','--out',str(target)]
    subprocess.run(cmd,check=True,timeout=240,env=env)
    result=json.loads((target/'metrics.json').read_text())
    assert result['n_test']>0
    assert result['n_trials']==7
    assert len(result['models'])==7
    subprocess.run([sys.executable,'-m','src.report','--results',str(target)],check=True,env=env)
    subprocess.run([sys.executable,'-m','src.analysis','--results',str(target), '--bootstrap','20'],check=True,env=env)
    assert (target/'report.html').exists()
    assert (target/'analysis.json').exists()
