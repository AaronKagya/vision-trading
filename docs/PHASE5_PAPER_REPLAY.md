# Phase 5 — Historical paper replay v0.3

This is a **historical research demonstration**, not a live paper-trading brokerage account, a real-time model, a profitable strategy or a trading recommendation.

## Evidence and methodology

- Source forecast predictions: immutable GitHub Actions artifact for the [locked January–March 2025 holdout](https://github.com/AaronKagya/vision-trading/actions/runs/42043691500). Trained on 2024; the test data were not used for fitting.
- One-hour forecasting horizon; observations produced every 15 minutes are **overlapping**. The replay selects one prediction every four rows to form **2,159 disjoint one-hour intervals**.
- Positions: `+1` when P(up) ≥ 0.55; `−1` when P(up) ≤ 0.45; otherwise `0` (abstain). This rule was defined before the 2025 Q1 holdout, but both periods have now been examined. No further validation claims can rely on those same periods being untouched.
- Illustrative log-P&L proxy per interval: position × one-hour forward log-price return − 0.0005 × absolute position turnover. Starting flat and flattening at the end count toward transaction costs.
- Verified result from Q1 archived artifact: **net log-P&L proxy −0.30326038812017153**, 524 active intervals, 536 units of turnover. Not a compounded investment return. No bid/ask, real spread, slippage, leverage, borrowing, fill or liquidation modeling.
- Phase 4 [epoch audit](https://github.com/AaronKagya/vision-trading/actions/runs/42226885906): 2024 validation selected a maximum 20-epoch cap; the external April–June 2025 evaluation obtained 50.81% mean LSTM accuracy vs 50.94% baseline, with negative net proxies across all seeds. No profitability demonstration.

## Rebuilding the page from official results

Download the unaltered ZIP artifacts from the two linked GitHub Actions runs above:

- `Vision_Trading_2025Q1_Holdout_Results.zip` (the locked Q1 prediction artifact)
- `Vision_Trading_Phase4_selection.zip` (the five-seed 2024 validation selection artifact)
- `Vision_Trading_Phase4_holdout.zip` (the 2025 Q2 external scoring artifact)

Then from the repository root:

```bash
python -m scripts.build_paper_replay \
  --q1-holdout-zip path/to/q1.zip \
  --epoch-selection-zip path/to/selection.zip \
  --epoch-holdout-zip path/to/q2.zip \
  --out site/paper/index.html
python -m pytest -q tests
```

The builder rejects missing prediction arrays, invalid scores, inconsistent timestamps, mismatched policy costs or P&L, and invalid Phase 4 study provenance. A static copy of the finished HTML is committed for offline viewing. Rebuilding the page requires no exchange account or API key.

## Research roadmap

1. Public historical replay and verified research dashboards (this milestone).
2. **Forward-only paper monitoring** with a persistent ledger, model versioning, an independently established dataset cutoff and explicit broker assumptions; no real orders.
3. Cost/risk stress tests and multiple independent periods/assets, including separately researched Volatility 75 Index (deferred).
4. Real-money execution **only** after adequate evidence and operational/regulatory review. We currently have no evidence to justify live autonomous trading.
