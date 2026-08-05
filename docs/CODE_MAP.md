# Code Map

## Theory and Sensitivity

| File | Role | Portability |
|---|---|---|
| `research_code/theory_and_sensitivity/MeanImped.py` | Equivalent-impedance model for a uniform labyrinth channel | Core class is portable; used by the example runner |
| `research_code/theory_and_sensitivity/ChangeImped.py` | Equivalent-impedance model for variable channel widths | Core class is portable; historical `main` uses an absolute data path |
| `research_code/theory_and_sensitivity/2.4.1 LatinHypercube.py` | LHS generation | Uses included local sample files |
| `research_code/theory_and_sensitivity/2.4.2 sobol灵敏度分析.py` | Sobol sensitivity analysis | Historical absolute paths must be replaced |
| `research_code/theory_and_sensitivity/2.4.3随机森林重要性排序.py` | Random-forest and permutation importance | Historical absolute paths must be replaced |

The remaining files in this directory are plotting, validation, and exploratory scripts retained from Chapter 2.

## Forward Design

| File | Role | Portability |
|---|---|---|
| `research_code/forward_design/3.1SparseSample.py` | Sparse structural sampling | Writes relative output files |
| `research_code/forward_design/3.1.4_re_comsol.py` | Merge or filter rerun COMSOL cases | Caller supplies filenames; COMSOL exports required |
| `research_code/forward_design/3.2.1_Extract_peak_hz.py` | Extract peak frequency and absorption | Input directory must be configured |
| `research_code/forward_design/3.3.2SurrogateModel.py` | LightGBM/SVR surrogate experiments | Several historical absolute paths remain |
| `research_code/forward_design/CnnFc.py` | CNN forward surrogate model | Full training arrays are external |
| `research_code/forward_design/3.3CNN_0.9988.ipynb` | Notebook for the reported CNN experiment | Kernel metadata records Python 3.10.13 |
| `research_code/forward_design/12.py` | Forward-model validation and plotting | Included arrays cover only selected plotting cases |

Historical FNN/LSTM/CNN-LSTM/TCN comparison files were moved to `research_code/legacy/forward_baselines/`.

## Inverse Design

| File | Role | Portability |
|---|---|---|
| `research_code/inverse_design/preprocessing/extract_absorb_para.py` | Assemble absorption/parameter matrices from COMSOL exports | Directory arguments are embedded in historical calls |
| `research_code/inverse_design/preprocessing/extract_peak_W.py` | Extract peak and geometry features | COMSOL exports required |
| `research_code/inverse_design/preprocessing/extract_virtual.py` | Assemble virtual-design training rows | Historical absolute paths remain |
| `research_code/inverse_design/functional_network/FCmodel.py` | Functional design network | Main entry expects external 9-column matrices |
| `research_code/inverse_design/functional_network/PS0_pyswarm4.4.py` | PSO-assisted coupling optimization | Requires restored model inputs and checkpoint |
| `research_code/inverse_design/geometry_network/inverseCnn.py` | Inverse geometry network | Original `test()` has a constructor mismatch and missing historical filenames |
| `research_code/inverse_design/geometry_network/TCN.py` | Temporal convolution experiment | Model definition only |

## Legacy Experiments

| Path | Reason retained | Status |
|---|---|---|
| `research_code/legacy/forward_baselines/` | FNN, LSTM, CNN-LSTM, and TCN comparison architectures | Provenance only; historical paths and checkpoint names remain |
| `research_code/legacy/forward_pso/PS0_pyswarm.py` | Prototype connecting the forward CNN to PySwarms | Incomplete: `__main__` optimizes a demonstration function and `model_fun()` normalization is unverified |

Tutorial-only PSO code, random-data decision-tree code, incomplete formula drafts, exact duplicates, empty scripts, and the resulting blank tree image were not included in the curated archive. See `docs/EXCLUSIONS.md` for the file-by-file rationale.

## Provenance Policy

Source files are copied without changing scientific formulas, model topology, or training logic, except for the narrowly documented conflict, import, and string-literal fixes in `CHANGELOG.md`. Portable wrappers and future cleanup should be added separately so the historical record remains auditable. Exact duplicate source files, empty scripts, caches, and explicitly third-party code were removed during curation.
