import io,json,zipfile
from pathlib import Path
import numpy as np
import pytest
from scripts.analyze_abstention import analyze

def create(tmp_path,seed,shift=0):
    dest=tmp_path/f'btcusdt_2024_seed{seed}.zip'
    y=np.array([0,1,1,0,1,0],dtype=int)
    fwd=np.array([.01,-.01,.03,-.02,.01,.01])
    bio=io.BytesIO()
    np.savez_compressed(bio,te=np.arange(6),t_idx=np.arange(20,26),y=y,fwd=fwd+shift,
                        p_lstm_ohlc=np.array([.4,.6,.8,.2,.5,.5]))
    with zipfile.ZipFile(dest,'w') as z:
        z.writestr(f'results_seed_{seed}/preds.npz',bio.getvalue())
        z.writestr(f'results_seed_{seed}/metrics.json',json.dumps({
            'config':{'seed':seed,'horizon':2,'cost':.0005},
            'data_provenance':{'sha256':'same-hash'},'n_test':6}))
    return dest

def test_reconstruct_five():
    import tempfile
    with tempfile.TemporaryDirectory() as x:
        p=Path(x);files={i:create(p,i) for i in range(5)}
        result=analyze(files)
        assert len(result['seeds'])==5
        assert result['aggregate']['coverage_mean']>0

def test_detect_different_returns():
    import tempfile
    with tempfile.TemporaryDirectory() as x:
        p=Path(x);files={i:create(p,i,shift=(.01 if i==4 else 0)) for i in range(5)}
        with pytest.raises(ValueError):analyze(files)

def test_reject_missing_seed():
    with pytest.raises(ValueError):analyze({})
