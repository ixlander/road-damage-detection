New-Item -ItemType Directory -Force -Path data\raw

kaggle datasets download `
  -d aliabdelmenam/rdd-2022 `
  -p data\raw `
  --unzip