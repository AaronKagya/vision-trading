"""One-shot genuine five-checkpoint observational inference; no live orders."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.forward_ai import (get_public_klines, verified_checkpoint_manifest,
                            verified_models, predict_frozen_ensemble,
                            advance, public_summary)


def run(path, root='models/phase4_2024'):
    output=Path(path)
    state=json.loads(output.read_text(encoding='utf-8')) if output.is_file() else None
    manifest=verified_checkpoint_manifest(root)
    history,observed_ms=get_public_klines()
    def score(data):
        models=verified_models(root)  # load only when an eligible, new candle is present
        return predict_frozen_ensemble(data, models)
    result=public_summary(advance(state, history, observed_ms, manifest, score))
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.with_suffix('.tmp')
    staging.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    staging.replace(output)
    print(f"Shadow model: recorded={result['metrics']['forecasts']} settled={result['metrics']['settled']}; status={result['status']}",flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',default='published/ai/state.json')
    parser.add_argument('--model-root',default='models/phase4_2024')
    args=parser.parse_args()
    run(args.state,args.model_root)
