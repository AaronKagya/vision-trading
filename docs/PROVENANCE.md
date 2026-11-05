# Provenance and handling

- Reference project: https://github.com/AaronKagya/vision-trading
- Specific baseline: `vision-trading (new).zip` on branch `main`, GitHub content blob SHA `fe53a056ddbeef41485dab72641d41358ea68d4f` (as observed during this review).
- Historical evidence: original user-uploaded `metrics.json`, copied byte-for-byte into `historical_evidence/metrics_seed0_original.json`.
- This corrected source tree is committed on the `fix/vision-trading-phase1-corrected` branch, while the original baseline archive remains preserved on `main`.
- Source is a research study, not a financial product. The original experiment source CSV, trained weights and `preds.npz` were not accessible as part of the supplied ZIP or upload.
- **Important:** Any corrected results should be compared only after controlling data version, evaluation boundaries, model seed and updated source code. The input archive's model and chart-renderer corrections invalidate a direct expectation of bitwise identical numerical forecasts.
