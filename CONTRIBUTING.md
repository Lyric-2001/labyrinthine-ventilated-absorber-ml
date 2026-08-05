# Contributing

This repository currently preserves research provenance. Before proposing a change:

1. Open an issue describing the affected experiment and expected scientific behavior.
2. Do not silently replace original scripts, datasets, checkpoints, or reported metrics.
3. Put portable interfaces and cleanup work in new files when changing the original code would obscure provenance.
4. Add a focused validation or test for changes to formulas, preprocessing, network dimensions, or optimization objectives.
5. Run `python tools/verify_repository.py --strict` before submitting a pull request.

Large generated data and COMSOL models must not be committed directly. Publish them through an approved research-data service and add their persistent URL and SHA-256 checksum to `docs/DATASETS.md`.

The repository does not yet grant an open-source license. External contributions should wait until the rights holders approve a contribution and licensing policy.
