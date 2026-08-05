# Data Directory

- `samples/` contains compact inputs that can be stored directly in Git.
- `validation/` contains selected COMSOL and optimization outputs used for visual comparison.
- Full training matrices are excluded and listed with hashes in `../docs/DATASETS.md`.

Text files are whitespace-delimited unless the extension is `.csv`. COMSOL exports generally require `skiprows=5`.
