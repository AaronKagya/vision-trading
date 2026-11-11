# Phase 2: controlled representation experiment

**Status (2026-11-05): implemented and synthetically smoke-tested; real-data performance NOT evaluated.**

## Objective

Test whether a single-frame chart CNN combined with a numerical OHLC LSTM provides a better predictive representation than either the original numerical LSTM or the original visual world model. Both branches see **the same historical OHLC information**. The chart cannot introduce additional raw information.

### Predeclared experiment

- Asset: BTCUSDT spot, 15-minute OHLC. Keep the source file and its SHA-256.
- Prediction label: the sign of the close-to-close log return, **four 15-minute candles ahead**.
- Chart encoding: 32 candles per trailing 15m/1h/4h view, 64x64 pixels, no future candles.
- New **fusion** candidate: 48-unit numerical LSTM, 64-dimensional chart CNN and a 48-unit fusion head, dropout 0.1. Image and numerical inputs use candles at or before prediction time `t`.
- Four expanding walk-forward folds, with embargo, validation-only early stopping and training-only scaling.
- Baseline seven models remain untouched in identity, making the fusion candidate the eighth tracked model. Any subsequent tuning, additional model families, image backbones, horizons, filter thresholds or assets used for choosing the winner must be recorded as trials.
- Primary criterion: *paired out-of-sample accuracy difference versus `lstm_ohlc`*, with 95% moving-block bootstrap CI. Positive point estimate alone is **not** sufficient; even a positive CI is not proof of economically profitable trading.
- Secondary criteria: AUC, Brier, ECE, net log-P&L **proxy**, PSR, DSR. Compare fusion against world model, and inspect per-fold performance.
- Confidence intervals and confidence bins are diagnostic, not an authorization to optimize on the held-out test. Use validation or a wholly new untouched period for threshold selection.

## Execution on a GPU machine with the historical CSV

Run the commands from the repository root after installing `requirements.txt` (and optional encoders only if needed):

```bash
python -m pytest -q tests

# Obtain exact, legal, complete BTCUSDT_15m.csv separately;
# verify timestamp alignment and SHA-256 before comparing with previous work.
python -m src.run_experiments \
  --data data/BTCUSDT_15m.csv \
  --models phase2 --window 32 --img 64 --horizon 4 --K 6 \
  --folds 4 --epochs 10 --bs 128 --encoder cnn --lr 0.001 \
  --lam 0.5 --cost 0.0005 --bootstrap 1000 --seed 0 \
  --save-checkpoints --out results_phase2

python -m src.report --results results_phase2
python -m src.phase2_eval --results results_phase2 \
  --data data/BTCUSDT_15m.csv --bootstrap 1000

# Repeat fixed experiment with different neural random seeds, without
# selecting a winning seed on test performance.
python run_seeds.py --data data/BTCUSDT_15m.csv --models phase2 \
  --window 32 --img 64 --horizon 4 --K 6 --folds 4 \
  --epochs 10 --bs 128 --seeds 0 1 2 3 4
```

The original seven-model experiment can still be run with `--models all` (default). `--models phase2` requests all **eight** configurations. It is normal for the revised baseline results to differ from an earlier version whose code contained defects.

## Short offline verification, not a trading performance evaluation

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m src.run_experiments \
  --data synthetic_null --n-synth 680 --window 16 --img 32 \
  --K 2 --folds 1 --epochs 1 --bs 16 --bootstrap 20 \
  --models phase2 --out results_phase2_smoke
python -m src.report --results results_phase2_smoke
python -m src.phase2_eval --results results_phase2_smoke --bootstrap 20
```

`synthetic_null` tests software execution only, not market skill. **Do not compare its accuracy to the BTCUSDT results or describe it as profitability.**

## Interpretation guardrails

- The completed code is **not** a trading bot or live broker integration.
- A world-model/vision model can learn different structure, but the chart is a lossy deterministic transformation of numerical candles.
- DSR is an approximate selection-bias adjustment whose result is only as reliable as the total number of recorded trials.
- The trading outputs are log-P&L approximations without actual spreads, funding, financing, borrow, slippage, margin or liquidation. Do not apply them to live capital.
- A large reported accuracy improvement without out-of-period stability and real execution costs is insufficient evidence.

## Later: Volatility 75 Index

Volatility 75 is intentionally **deferred**, as requested. When ready, use a documented authorized source of timestamped OHLC data, verify its precise market timebase/candle duration and missing-data rules, and keep its experiments and trial ledger **separate** from the BTCUSDT research. In particular, do **not** reuse Binance spot funding, execution, or transaction-cost assumptions for a synthetic index. Do not calibrate the model on Volatility 75 and then report the same period as untouched out-of-sample evidence.
