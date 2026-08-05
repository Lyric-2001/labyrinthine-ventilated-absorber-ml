# Curation Exclusions

The parent thesis workspace remains untouched. The files below were deliberately left out of this GitHub-ready archive because they are tutorial demonstrations, incomplete drafts, duplicate variants, or blank outputs rather than reliable research entry points.

| Original curated-copy path | Decision | Reason |
|---|---|---|
| `research_code/forward_design/PS0_code.py` | Excluded | Generic two-variable PSO teaching example; it does not load absorber data or a research model |
| `research_code/forward_design/tree_picture.py` | Excluded | Fits a decision tree to random data; it is unrelated to the reported LightGBM experiment |
| `research_code/forward_design/cnn_lstm.py` | Excluded | Unresolved duplicate/variant with no unique result or checkpoint mapping |
| `research_code/inverse_design/preprocessing/4-12.py` | Excluded | Short plotting/helper draft with only historical absolute paths |
| `research_code/inverse_design/preprocessing/4.3-2.py` | Excluded | Exploratory helper tied to unavailable files; no unique reproducible output identified |
| `research_code/inverse_design/preprocessing/PSO.py` | Excluded | Generic/incomplete preprocessing experiment, not the functional-network optimization entry point |
| `research_code/theory_and_sensitivity/11.py` | Excluded | Variable-count mismatch with its accompanying test data; no identifiable thesis result |
| `research_code/theory_and_sensitivity/2.4.1RBD_fast.py` | Excluded | Synthetic RBD-FAST demonstration rather than an analysis of the curated study outputs |
| `research_code/theory_and_sensitivity/ChangeImpedTWO.py` | Excluded | Incomplete alternative implementation with no returned result from its main calculation |
| `results/figures/forward_design/lightgbm-tree.jpg` | Excluded | 640 x 480 blank image; the generating script used random data |

The five FNN/LSTM/CNN-LSTM/TCN comparison implementations and the forward PSO prototype are retained under `research_code/legacy/` with explicit warnings. The excluded files are still available in the original parent workspace if a later archival release requires them; they were not deleted from the source workspace.
