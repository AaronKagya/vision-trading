# Do Visual World Models Add Information Beyond Numerical Features?

Leakage-free test of chart-image forecasting against numeric baselines, with overfitting-aware statistics.
Historical data only. No trading, no accounts, no real money. Not investment advice.

## What it does
1. **Data**: free Binance history (15-minute candles) or built-in synthetic data for testing.
2. **Charts**: each moment becomes a 64x64 image with three timeframes (15m, 1h, 4h) in the colour channels. Built only from candles up to that moment; `tests/test_no_lookahead.py` proves it.
3. **Models** (7, all compared on the same unseen periods):

| Name | What it is | Role |
|---|---|---|
| majority, persistence, logreg_lags | trivial and linear baselines | floor |
| lstm_ohlc | LSTM on the same OHLC numbers the charts show | the numeric competitor (key test) |
| cnn_static | CNN on one chart | ablation: no world model |
| wm_nolatent | CNN + GRU over 6 consecutive charts, direction loss only | ablation: no latent prediction |
| **wm** | CNN + GRU, predicts the *next chart's latent vector* plus direction | the visual world model |

4. **Evaluation**: expanding walk-forward folds with an embargo, no shuffling; accuracy, AUC, Brier, calibration (ECE); strategy with 5 bps costs; probabilistic and deflated Sharpe ratio; block-bootstrap confidence intervals; paired test of each visual model against the LSTM.
5. **Outputs**: `results/report.html` (shareable results page) and `app.py` (interactive demo).

## Run it
```bash
pip install -r requirements.txt
python -m tests.test_no_lookahead                                  # must print PASS

# 1) Download data (check Binance's terms first). ~1 year of BTCUSDT, 15m
python -c "from src.data import download_binance as d; d('BTCUSDT','15m','2024-01','2024-12')"

# 2) Pipeline smoke test on synthetic data with a planted signal (minutes on CPU)
python -m src.run_experiments --data synthetic_planted --n-synth 4000 --folds 3 --epochs 4 --out results_test
python -m src.report --results results_test

# 3) Real run (use a GPU, e.g. free Colab: Runtime > Change runtime type > T4)
python -m src.run_experiments --data data/BTCUSDT_15m.csv --epochs 10
python -m src.report
streamlit run app.py
```
On a CPU the real run is slow (hours). On a free Colab GPU it should take minutes to tens of minutes.
Encoders: `--encoder cnn` (default), `--encoder resnet18 --pretrained`, `--encoder vit_tiny --pretrained` (needs torchvision/timm and internet for weights).

## Rules that keep the result honest
- Every configuration you try counts. If you tune anything (window, horizon, epochs, encoder), rerun with `--extra-trials N` so the deflated Sharpe knows. Keep a list of what you tried.
- Do **not** choose settings by looking at the test results. Choose them on validation data, then run the final test once.
- Report the null result if that is what you get. The report writes an honest verdict automatically.
- `synthetic_null` (no signal) should give about 50% accuracy; `synthetic_planted` should give clearly more. If null looks good, something leaks.

## Six-week plan (matches your mid-semester page)
| Week | Do | Done when |
|---|---|---|
| 1 | Download data, run the look-ahead test, run the synthetic smoke test, start the literature review | PASS printed; report.html opens |
| 2 | Real data run with baselines only; choose horizon and window on validation | baseline table filled |
| 3 | Encoders (cnn, resnet18, vit_tiny) on the real data | encoder comparison |
| 4 | Full world-model run, ablations | `results/` for the final configuration |
| 5 | Volatility-split error analysis, robustness on a second coin or period, count all trials | extra tables |
| 6 | Poster from report.html figures, demo with app.py | poster, demo, README |
