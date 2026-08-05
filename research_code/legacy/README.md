# Legacy Research Scripts

This directory contains experiment variants that are related to the thesis but are not reliable reproduction entry points.

- `forward_baselines/` preserves FNN, LSTM, CNN-LSTM, and TCN comparison implementations. They retain historical data paths and checkpoint names.
- `forward_pso/PS0_pyswarm.py` is an unfinished prototype connecting the forward CNN to PySwarms. Its `__main__` block still optimizes a demonstration function, and the scaling in `model_fun()` is not validated.

The files are included to preserve research provenance. Do not use their reported behavior without first restoring the matching data, checking normalization, and validating the objective against COMSOL results.
