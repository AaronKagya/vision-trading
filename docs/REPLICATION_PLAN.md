# Corrected experiment: recommended next actions

**No evidence of profitable live trading skill has been established.** Retain all existing historical outputs and use a new results directory for every corrected experiment.

1. Obtain and independently verify the exact BTCUSDT 15m historical CSV; record download source, SHA-256, timestamps, row count, and missing intervals.
2. Run all automated tests. Preserve the resulting test log and Python/PyTorch versions.
3. Freeze research hypotheses and primary configuration *before* looking at corrected test results. Document every encoder, window, horizon, cost or threshold explored.
4. Re-run the four-fold comparison in `results_corrected_seed0`. Do not overwrite historical seed-0 JSON; the corrected causal model and image renderer have different semantics.
5. Save prediction arrays, fold boundaries, checkpoints, and summary metrics. Audit reported accuracy, paired block-bootstrap intervals and calibrated probabilities.
6. Repeat *unchanged* config across five random seeds using `run_seeds.py`; report mean/SD and how often a visual model exceeds the numeric baseline.
7. Repeat on ETHUSDT and/or a later pre-registered date window. Avoid choosing an asset based on favorable holdout performance.
8. Evaluate sensitivity to realistic trade costs, including spreads, slippage, borrow/funding and short-sale feasibility; maintain separate forecasts and execution evaluations. Do not call synthetic log-P&L a tradable return.
9. Use forward paper trading only after the above. Monitor probability calibration, drift, latency, data outages and risk limits.
10. Update the research manuscript with the exact corrected run: authorship, study dates, actual data coverage, per-seed tables, code/data hashes, and limitations. Clearly state when the null hypothesis cannot be rejected.

**Optional deployability path:** after forward verification, wrap model inference in a stateless REST service with immutable training artifacts, monitoring, and audit logs. This repository intentionally does not place real orders or manage exchange secrets.
