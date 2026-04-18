# Architecture

The codebase is split into app entrypoints and reusable domain modules.

- `apps/`: runtime entrypoints for API, demo, and CLI
- `src/road_damage/common`: constants, config, IO validation
- `src/road_damage/inference`: shared model service, postprocessing, API schemas
- `src/road_damage/data`: dataset contracts, conversion, validation
- `src/road_damage/modeling`: train/evaluate/predict wrappers and metrics output

Key invariants:
- class map is fixed (`0 crack`, `1 pothole`)
- API/demo/CLI share one inference path
- model loading is controlled by model registry IDs
