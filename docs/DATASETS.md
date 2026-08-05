# Data and Large Files

## Included Data

The repository contains small inputs and representative validation results that are useful for inspection and plotting:

| Directory | Contents |
|---|---|
| `data/samples/theory/` | LHS and Sobol samples, impedance curves, and small Chapter 2 validation arrays |
| `data/samples/forward_design/` | 1,200 training geometries, 150 test geometries, COMSOL input tables, and CNN validation arrays |
| `data/validation/functional_network/` | PSO histories, baseline curves, and selected coupled-response exports |
| `data/validation/geometry_network/` | Six COMSOL/target validation curves for inverse-design checks |

Key included array shapes:

| File | Shape | Notes |
|---|---:|---|
| `samples/forward_design/LHS_train_sample.txt` | 1200 x 5 | LHS structural parameters |
| `samples/forward_design/LHS_test_sample.txt` | 150 x 5 | Held-out structural parameters |
| `samples/forward_design/origin_data/train_comsol_inputdata.txt` | 1200 x 6 | Geometry plus COMSOL-facing index/value column |
| `samples/forward_design/3.3.3val.txt` | 2855 x 7 | Validation rows used by plotting scripts |
| `samples/forward_design/cnn-validation-output.txt` | 2855 | Forward CNN predictions retained under a descriptive filename |
| `samples/theory/sobol_sample.txt` | 1024 x 7 | Sensitivity-analysis sample matrix |
| `validation/geometry_network/*.txt` | 571 x 2 | COMSOL text exports after five header rows |

COMSOL curve exports usually contain five metadata lines followed by frequency and absorption-coefficient columns. Historical scripts therefore call `numpy.loadtxt(..., skiprows=5)`.

## Full-Matrix Families

The workspace contains several matrix formats from different experiment stages. They are not interchangeable. The column splits below come from the surviving source code; a semantic name for every column was not found and should be added before claiming full reproducibility.

| Original files | Columns | Source-code interpretation | Use |
|---|---:|---|---|
| `inverse_model1/latin_{train,test}_absor_para.txt` | 9 | First 8 columns are inputs; final column is the target in `FCmodel.py::FcPredict()` | Primary functional-network matrix family |
| `inverse_model2/Latin_{train,test}_absorpara_model2.txt` | 14 | First 9 columns are curve/performance inputs; final 5 are geometry targets in `inverseCnn.py::test()` | Primary geometry-network matrix family |
| `闫龙辉/graduation_project/chapter_4/inverse_model1/{train,test}_para_absor.txt` | 9 | Earlier functional-network variant | Historical alternative; do not mix with the primary family |
| `闫龙辉/graduation_project/chapter_4/inverse_model2/{train,test}_absor_para.txt` | 16 | Earlier geometry-network variant | Historical alternative; do not mix with the primary family |
| `闫龙辉/graduation_project/chapter_4/inverse_model1/latin_test_absor_para_nonel_r.txt` | 7 | First 2 columns are inputs and final 5 are targets in the surviving `inverseCnn.py` main block | Test-only variant; matching training file was not found |
| `闫龙辉/graduation_project/chapter_4/train_para_absor.txt` | 577 | Wide parameter-plus-curve matrix | Historical derived matrix |

The two primary families above are the recommended starting point when the external data are archived. Before training, verify row counts, units, frequency ordering, parameter bounds, normalization statistics, and the exact checkpoint/source pairing.

## Available Local Files Kept Outside Git

The following original files are needed for full regeneration but are intentionally not copied into Git history. Paths are relative to the original working folder. SHA-256 hashes allow future archived copies to be verified.

| Original path | Bytes | SHA-256 |
|---|---:|---|
| `model/3.5.5自动化建模通风圆盘_宽带吸声.mph` | 910321662 | `D2002397292741E16DB6A4FD0549667CAC3D461FBCCD1E2A6FF9661A77218B80` |
| `inverse_model1/latin_train_absor_para.txt` | 76354457 | `2DD0FF804F3C58938DF16EC47A7CCF9DDD8BF9534C579AAB1B4E46148C0BA1F0` |
| `inverse_model1/latin_test_absor_para.txt` | 9550945 | `4869A95B04259D9AC8C5EC9BA7748E9CF31B50D35DD516837740A5417F0B5AFF` |
| `inverse_model2/Latin_train_absorpara_model2.txt` | 114725657 | `C182ED9306788B962F2BF03936EB9ECDFA3225CE23622EF25A60D7CEF6669047` |
| `inverse_model2/Latin_test_absorpara_model2.txt` | 14347345 | `339E2D0A510A843C0EA929EC32C265D61E7AA794801FC4ADA4BD4167FD04F6FD` |
| `闫龙辉/graduation_project/chapter_4/inverse_model1/train_para_absor.txt` | 63404177 | `B888ABB19572F8B2DB1E9FC0F361760855AE593B9B2E37B7C6DB97A5942D7A88` |
| `闫龙辉/graduation_project/chapter_4/inverse_model1/test_para_absor.txt` | 7924737 | `5E4543620EA22184D6D3F919DC43050EE5E3C23F12B2012BF3A66C85D5117E9B` |
| `闫龙辉/graduation_project/chapter_4/inverse_model1/latin_test_absor_para_nonel_r.txt` | 7268658 | `CDE23C8A26AC37B5AB0DF77A47576E93D8F30F811620E9051FFE952F4D56EEB7` |
| `闫龙辉/graduation_project/chapter_4/inverse_model2/train_absor_para.txt` | 133178664 | `B26DC508AA841D864C4E894087230AB10CF577F632100EE066AF70685A189D7E` |
| `闫龙辉/graduation_project/chapter_4/inverse_model2/test_absor_para.txt` | 16655898 | `70CA2C3A4AAC6C3B420C4F01416D23085A5ABD2E422A7BB6BEACFCB431406BC8` |
| `闫龙辉/graduation_project/chapter_4/virtual_data/train_virtual_absor_para.txt` | 55809519 | `94E949FCEEE464C4C4CCF53853237F84D9EACD6967C646D9FCA73AC937788632` |
| `闫龙辉/graduation_project/chapter_4/virtual_data/virtual_train.txt` | 9779400 | `DD0B926BC8A096F42DF418FD09E52C070F01C0CB293EAE4A90BFC23CEA50B677` |
| `闫龙辉/graduation_project/chapter_4/virtual_data/virtual_y.txt` | 8566267 | `ABA3CB3E5430334EEC67037DF8E1AD069627F2CCEA2E85676B423E5AF7D2D154` |
| `闫龙辉/graduation_project/chapter_4/train_para_absor.txt` | 5544445 | `2C6B6F588551BAF09B1E348B14A0EB8B1BA3D3A18B2234B932F6D24DD1168EE5` |

The machine-readable version is `docs/local-files-manifest.csv`.

## Referenced Inputs Not Found

The following names are referenced by retained scripts but were not found as standalone files in the supplied workspace:

- `train_ParaHzAbsorb.txt` and `test_ParaHzAbsorb.txt`
- `train_ParaPeakHz.txt` and `test_ParaPeakHz.txt`
- `rand_val_hz.txt`
- `latin_train_absor_para_nonel_r.txt`

Some equivalent information may be derivable from the available full matrices or COMSOL exports, but no deterministic derivation script and schema were identified. They are therefore documented as missing rather than silently substituted. Historical plotting scripts also reference many experiment-specific COMSOL curves through absolute paths; the included validation files are representative, not an exhaustive mirror of those directories.

## Why These Files Are External

- GitHub rejects ordinary Git objects above 100 MiB and warns above 50 MiB.
- Large text matrices create noisy repository history and are generated artifacts rather than hand-authored source.
- The COMSOL file requires commercial software and licensed modules.
- A research-data repository provides stable identifiers, checksums, and better download behavior than Git history.

Recommended publication order:

1. Package the two primary matrix families and the COMSOL model as versioned archives; keep historical alternatives in a separately labeled archive.
2. Upload them to Zenodo or an institutional repository.
3. Add a data dictionary with units, column order, shapes, frequency grid, and train/test provenance.
4. Record the data DOI, archive filenames, sizes, and SHA-256 hashes here.
5. Keep the existing local filenames or document a deterministic rename map.
6. Do not place the files directly in `main` even if Git LFS is available, unless long-term LFS quota and access are confirmed.

## Excluded Non-research Material

MNIST, manuscripts, review files, Word/PDF/PPTX/VSDX/ZIP artifacts, old Git history, IDE settings, caches, and code or documents attributed to other researchers are not part of this repository.
