# Road Damage Detection (Crack and Pothole)

This repository provides a safer, modular PoC for road-damage detection with shared inference logic across API, demo, and CLI.

## Class mapping invariant

- `0 -> crack`
- `1 -> pothole`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt -c constraints.txt
```

## Project layout

```text
apps/
  api/main.py
  demo/app.py
  cli/infer_video.py
src/road_damage/
  common/
  data/
  inference/
  modeling/
configs/
  data/
  train/
  eval/
tests/
  unit/
  integration/
  regression/
```

Legacy paths remain for compatibility:
- `api/main.py`
- `demo/app.py`
- `src/infer_video.py`

## Data prep

```bash
python datasets/make_2class_dataset.py
python -m road_damage.data.validate --root data/rdd2class_yolo --report outputs/dataset_validation_report.json
```

## Train / eval / infer

Train wrapper (metadata + reproducibility lineage):
```bash
PYTHONPATH=src python -m road_damage.modeling.train --config configs/train/baseline.yaml --out outputs/train
```

Evaluate wrapper:
```bash
PYTHONPATH=src python -m road_damage.modeling.evaluate --config configs/eval/default.yaml --out outputs/eval
```

Video inference CLI:
```bash
python apps/cli/infer_video.py --source data/sample_video.mp4 --model_id default --out_dir outputs
```

API:
```bash
uvicorn apps.api.main:APP --host 127.0.0.1 --port 8000
```

Demo:
```bash
streamlit run apps/demo/app.py
```

## Reproducibility workflow

- Use config files in `configs/train` and `configs/eval`.
- Set explicit seed in train config.
- Persist `train_metadata.json` (commit SHA, config snapshot, dataset manifest hash).
- Keep dataset validation and manifest checks in CI for dataset-touching changes.

## Compatibility matrix

- Python: 3.10–3.11
- CUDA: optional, CPU supported by default configs
- Ultralytics: pinned via `constraints.txt` (`8.3.120`)

## Quality and security gates

- `ci.yml`: lint, type-check, tests, smoke inference/eval
- `security.yml`: dependency vulnerability scan + secret scanning

## Migration notes

- New shared inference core lives under `src/road_damage/inference`.
- Public API now prefers `model_id`; direct arbitrary local `model_path` is rejected unless mapped in whitelist (`ROAD_DAMAGE_MODELS`).
- Existing `/predict_image` route is preserved as v1 compatibility alias.

## Rollback guidance

- If deployment issues occur, revert to previous commit/tag and keep old runtime entrypoints (`api/main.py`, `demo/app.py`, `src/infer_video.py`) which are retained.
- Disable new wrappers by running legacy commands against prior revision while investigating config or registry mismatches.
