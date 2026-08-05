# Model Artifacts

This directory contains the original trained model files selected from the thesis workspace.

| Directory | Model family |
|---|---|
| `checkpoints/forward_design/` | CNN forward surrogate and LightGBM model |
| `checkpoints/functional_network/` | Functional design network variants |
| `checkpoints/geometry_network/` | Geometry inverse network variants |

## Provenance and Source Mapping

| Curated artifact | Original workspace name | Mapping status |
|---|---|---|
| `forward_design/CNN9988.pth` | `闫龙辉/graduation_project/chapter_3/CNN9988.pth` | Exact filename expected by `CnnFc.py`; training arrays remain external |
| `forward_design/lgbm_model.pkl` | `闫龙辉/graduation_project/chapter_3/lgbm_model.pkl` | Exact filename expected by `3.3.2SurrogateModel.py` |
| `functional_network/model_0/Lstm.pth` | `inverse_model1/model_0/Lstm.pth` | Exact path expected by `FCmodel.py` |
| `functional_network/model_1/Lstm0.9997.pth` | `inverse_model1/model_1/Lstm0.9997.pth` | Exact experiment file; referenced as an alternative in `FCmodel.py` |
| `functional_network/FCmodel-functional-design.pth` | `inverse_model1/4.2FCmodel (2).pth` | Renamed for clarity; no surviving script loads this original filename directly |
| `geometry_network/Latin_model2_0.9915.pth` | `inverse_model2/Latin_model2_0.9915.pth` | Byte-preserved; the retained `inverseCnn.py` expects another historical filename |
| `geometry_network/Latin_model2_0.9917_cnn2.pth` | `inverse_model2/Latin_model2_0.9917_卷积2.pth` | Filename normalized to ASCII; architecture pairing is unresolved |
| `geometry_network/Latin_model2_3.20_0.992_r0_cnn_2.pth` | `inverse_model2/Latin_model2_3.20_0.992_r0_cnn_2.pth` | Byte-preserved; architecture pairing is unresolved |
| `geometry_network/model_0/Latin_model2_3.19.pth` | `inverse_model2/model_0/Latin_model2_3.19.pth` | Byte-preserved; architecture pairing is unresolved |
| `geometry_network/model_1/Latin_model2_3.19.pth` | `inverse_model2/model_1/Latin_model2_3.19.pth` | Byte-preserved; architecture pairing is unresolved |

PyTorch and pickle files can execute deserialization logic. Load only trusted copies from this repository, verify `ARTIFACTS.sha256`, and use the pinned dependency versions. A matching filename is provenance evidence, not proof that the current script constructor and preprocessing reproduce the saved model. Validate a checkpoint against included reference curves before using it for optimization or reported metrics.
