# BPR ↔ ECDLP — FFT spectral analysis on toy curves

Skeptical empirical test of whether FFT/spectral analysis of BPR-style
"boundary phase fields" reveals usable information about the secret scalar
`k` in elliptic-curve discrete log problems on small synthetic curves.

**Constraint:** all experiments are confined to seeded toy curves and
synthetic test scalars. No real Bitcoin/Ethereum addresses, no leaked
nonce data, no secp256k1 attack code.

## Run

```bash
cd experiments/ecdlp_fft
python3 run_experiment.py --bits 12 14 16 18 --instances 30 \
    --keys-per-instance 1 --window-size 256 --outdir data
python3 summarize.py data/results.csv data
python3 distance_analysis.py data/results.csv
```

Total wallclock for the full run: ~83 s on a single CPU.

## Verdict

**No usable signal** — confirmed across two independent batteries:

1. **FFT-magnitude analysis** ([REPORT.md](REPORT.md)) — 0/200 buckets
   survive BH-FDR; smoking gun is that several "Q-coupled" BPR field
   constructions have provably Q-invariant magnitude spectra.
2. **Extended transform suite** ([REPORT_EXTENDED.md](REPORT_EXTENDED.md))
   — 39 features per trial (phase spectrum, autocorrelation, bispectrum,
   wavelet, multiscale entropy, Legendre walk, additive characters, graph
   Laplacian) plus MMD and group-aware classifier-AUC distinguishers.
   0/1560 marginal tests, 1/40 MMD, 0/40 AUC survive BH-FDR.

- 0 / 200 (bits, field, control) buckets survive BH-FDR at α=0.05
- Mutual information between spectral features and `k` at estimator floor
- No scaling improvement from 12-bit to 18-bit
- DDH framing: any positive result would have been a DDH distinguisher;
  none observed.
