## Correlated random walk

1. Compare the trajectories and autocorrelation patterns
2. Benchmark gradient flow

Contains 3 files:
- `corr.py`: benchmarks gumbel , cat
- `corr_scat.py` : benchmarks cat, cat++
- `corr_final.py` : benchmarks gumbel, cat, cat++

> The function `analyze_correlated_walks` has been commented out in all files. This function is used to compare trajectories and autocorrelation patterns.

To run the code, use the following command:
```bash
python corr_final.py
```
The results will be printed in the terminal.