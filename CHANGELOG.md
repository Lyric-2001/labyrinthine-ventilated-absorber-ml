# Changelog

All notable changes to this curated repository are documented here.

## 1.0.0 - 2026-08-05

- Organized thesis research code into theory, forward-design, and inverse-design workflows.
- Added GitHub-friendly sample data, validation curves, representative figures, and checkpoints.
- Added pinned Python 3.10 environment definitions and a portable theory example.
- Added formal article citation metadata and repository documentation.
- Excluded oversized datasets, the COMSOL model, manuscripts, caches, IDE files, MNIST, and third-party material.
- Removed unresolved Git conflict markers from the curated copy of `3.1.4_re_comsol.py` and retained its test-data branch as the default.
- Made Matplotlib optional when importing the curated `MeanImped.py`; its original plotting behavior remains available when run as a script.
- Marked two mixed-separator Windows path literals as raw strings to remove Python escape warnings without changing their values.
- Moved historical forward-model baselines and the incomplete forward PSO prototype under `research_code/legacy/`.
- Removed tutorial-only scripts and a blank random-data tree image from the curated archive.
- Documented incompatible matrix families, missing referenced inputs, and unresolved checkpoint mappings.
