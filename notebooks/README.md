# Model notebooks

Scratch space and reproducible experiments for the NomaApp crop-disease models.
Everything here is **training-side**; the serving code lives in `app/`.

```
notebooks/
├── 01_data_exploration.ipynb      # dataset sanity checks + class balance
├── 02_train_maize_baseline.ipynb  # transfer-learning baseline for one crop
├── 03_export_and_validate.ipynb   # .keras -> TFLite/ONNX + serving parity check
├── utils.py                       # shared conventions & helpers
├── requirements-notebooks.txt     # extra deps beyond ../requirements.txt
├── data/                          # datasets (git-ignored)
└── artifacts/                     # checkpoints, plots, exports (git-ignored)
```

## Quick start

```bash
# from the repo root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r notebooks/requirements-notebooks.txt
jupyter lab notebooks/
```

The notebooks are path-agnostic: launching Jupyter from the repo root *or*
from inside `notebooks/` both work. Datasets can live elsewhere by setting
`NOMA_DATA_DIR=/path/to/dataset`.

## Expected dataset layout

`tf.keras.utils.image_dataset_from_directory` format — one folder per class:

```
notebooks/data/<dataset>/train/<class>/img_001.jpg
notebooks/data/<dataset>/valid/<class>/img_002.jpg
notebooks/data/<dataset>/test/<class>/img_003.jpg   # optional
```

Class folder names do not have to match `CLASS_NAMES`, because every notebook
writes a `class_names.json` next to the weights. **Always ship that file with
the model** — an index-shuffled classifier is worse than no classifier.

## Ground rules (train/serve parity)

These are enforced by `utils.py` and mirrored in `app/services/preprocess.py`:

| Convention | Value |
| --- | --- |
| Input size | `224 × 224` |
| Colour | RGB |
| Scaling | `float32 / 255.0` |
| Batched shape | `(1, 224, 224, 3)` |
| Class order | `utils.CLASS_NAMES` ≡ `app/services/inference.py` |

If you deliberately change one of these, change it in the app code in the same
commit and note it in the PR. `03_export_and_validate.ipynb` runs a parity
check that catches most drift.

## Naming & handoff

Artifacts follow the pattern already used by the server:

```
artifacts/<crop>_<stage>_<version>.keras     # e.g. maize_expert_v1.keras
artifacts/<crop>_<stage>_<version>.tflite
artifacts/<crop>_<stage>_<version>.onnx
artifacts/<crop>_<stage>_<version>.class_names.json
```

When a model is promoted:

1. Move the weight file into `app/models/` and register it in
   `app/models/registry.py` (or, for the ONNX router/expert layout, publish it
   under `model_cache/` and add an entry to `model_cache/manifest.json`).
2. Bump `version` in the manifest — the API reports it.
3. Keep the `.class_names.json` alongside it.

`utils.manifest_entry()` generates a correctly-shaped manifest block.

## What not to commit

`data/` and `artifacts/` are git-ignored on purpose. Keep raw datasets and
weight files out of Git — share large files via Drive/S3 or a model registry.
Notebook outputs containing plots are fine, but strip huge HTML dumps before
committing (`jupyter nbconvert --clear-output --inplace <file>`).
