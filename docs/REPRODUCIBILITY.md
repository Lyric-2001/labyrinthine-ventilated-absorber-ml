# Reproducibility Notes

## What Works Immediately

After installing `requirements.txt`, the portable runner executes the original uniform-channel impedance class and writes a figure:

```bash
python examples/run_theoretical_model.py --output results/generated/theoretical-absorption.png
```

The runner always writes a CSV. It also writes the requested PNG when Matplotlib is available.

The repository checker uses only the Python standard library:

```bash
python tools/verify_repository.py --strict
```

Small LHS matrices, COMSOL curve exports, validation plots, and model checkpoints can be inspected directly.

## Historical Script Assumptions

The research scripts were produced during iterative thesis experiments. They commonly assume:

- execution from the script's own directory;
- fixed `C:/`, `E:/`, `F:/`, or `/kaggle/` data paths;
- data filenames that changed between experiments;
- COMSOL text exports with five header rows;
- CUDA device 0 when available;
- large arrays documented in `docs/DATASETS.md`;
- checkpoints loaded with architecture definitions from the corresponding script.

These assumptions are preserved rather than silently rewritten. The repository checker reports their locations as warnings.

## Suggested Reproduction Order

### 1. Theoretical model

Use `examples/run_theoretical_model.py` or import `Impedeance` from `MeanImped.py`. The default example uses the thesis parameters for a seven-channel labyrinth and scans 135-695 Hz.

### 2. Sampling and FEM data

Generate LHS/Sobol samples with the scripts in `theory_and_sensitivity/`, then run the COMSOL model externally. The original model metadata reports COMSOL 6.0.0.318 and the COMSOL, Acoustics, CAD Import, and Design products. Exact license availability must be checked on the target machine.

COMSOL output curves are consumed as two-column text after five metadata rows. Data extraction scripts are located in `forward_design/` and `inverse_design/preprocessing/`.

### 3. Forward surrogate model

Restore the forward training data expected by `3.3.2SurrogateModel.py` or `CnnFc.py`. The primary research notebook is `3.3CNN_0.9988.ipynb`; its saved metadata records Python 3.10.13. Keep NumPy below version 2.

### 4. Functional and inverse networks

Restore the external matrices listed in `docs/DATASETS.md`. `FCmodel.py` consumes a 9-column matrix and predicts the final target column. `inverseCnn.py` uses five structural outputs and expects its matching architecture/checkpoint pair.

The historical `inverseCnn.py::test()` reconstructs `CNN_FC` with a missing `inputsize` argument and references unavailable checkpoint names. Treat it as provenance code until the architecture/checkpoint mapping is reconciled against the original experiment notes.

### 5. PSO and broadband coupling

Run PSO only after validating a restored checkpoint against the included COMSOL curves. Historical optimization scripts import sibling model files, so run them from the corresponding source directory or convert the directories into packages in a future cleanup branch.

The forward-model PSO prototype is retained under `research_code/legacy/forward_pso/`. It is not a reproduction entry point: its `__main__` block runs a demonstration objective rather than `model_fun()`, and its input scaling has not been reconciled with the forward training matrix.

## Validation Targets Reported in the Study

- Functional design networks: maximum curve errors 0.032 and 0.058; average error approximately 0.0008.
- Inverse networks: reported MAE 0.0461/0.0493 and MAPE 0.0072/0.0077 for the two networks.
- Coupled design: absorption coefficient above 0.8 from 935 to 1110 Hz, average approximately 0.89.
- Within that range: 78 Hz above 0.9, average approximately 0.93.

Use the figures and validation text files as reference outputs, not as a substitute for a scripted metric test.

## Known Gaps Before One-command Reproduction

1. Publish the external datasets and COMSOL model with a stable DOI.
2. Define a schema for every full training matrix.
3. Replace embedded paths with command-line options or a shared configuration file.
4. Match each checkpoint to an immutable architecture identifier.
5. Add deterministic train/evaluate commands and metric regression tests.
6. Record operating system, CUDA, GPU, and COMSOL solver settings for the final published run.

Until those items are completed, describe the repository as a curated research archive with partial reproducibility rather than a fully automated replication package.
