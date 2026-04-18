# Release checklist

- [ ] CI (`ci.yml`) green
- [ ] Security workflow (`security.yml`) green
- [ ] Dataset validator report reviewed for release dataset
- [ ] `train_metadata.json` and evaluation outputs archived
- [ ] API compatibility verified (`/predict_image`, `/v1/predict_image`)
- [ ] Regression tests (golden inference + metric floors) passing
- [ ] Migration notes and rollback plan reviewed
