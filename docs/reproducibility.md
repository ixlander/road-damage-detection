# Reproducibility

Use config-driven entrypoints and preserve metadata artifacts.

## Train

`road_damage.modeling.train` stores:
- seed
- commit SHA
- config snapshot
- dataset manifest hash and class map

## Evaluate

`road_damage.modeling.evaluate` writes JSON/CSV outputs, including threshold sweep.

## Dataset lineage

- Validate dataset structure and labels with `road_damage.data.validate`
- Persist manifest JSON for source/version/hash lineage
