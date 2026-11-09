"""One-shot market observation and forward-only paper-ledger update.

A GitHub Actions cron repeats this program; it cannot place real orders.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.forward_paper import get_public_market_snapshot, advance, public_summary


def run(path: str) -> dict:
    target = Path(path)
    existing = json.loads(target.read_text(encoding='utf-8')) if target.is_file() else None
    candles, price, now_ms = get_public_market_snapshot()
    result = public_summary(advance(existing, candles, price, now_ms))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix('.tmp')
    staging.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    staging.replace(target)
    print(f'Paper observation saved; closed={result["metrics"]["closed_longs"]}, decisions={result["metrics"]["decisions"]}, '
          f'status={result["status"]}, price={price:.2f}, observed_ms={now_ms}', flush=True)
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state', default='published/live/state.json')
    a=ap.parse_args()
    run(a.state)
