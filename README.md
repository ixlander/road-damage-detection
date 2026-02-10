# Road Damage Detection (Crack and Pothole)

This repository contains an end-to-end computer vision project for road damage detection using YOLOv8.
It includes:
- Dataset preparation (class merging into 2 classes)
- Training (smoke test and full training command)
- Video inference script (annotated MP4 + CSV predictions)
- FastAPI inference service (JSON + annotated image)
- Streamlit demo (image and video inference UI)

## Classes

The project uses 2 classes:
- 0: crack
- 1: pothole

The original dataset distribution was merged into these two classes. In the provided Kaggle split, `class_id 4` corresponds to pothole; all other class IDs are merged into crack.

## Dataset

Source: RDD2022 from Kaggle (already in YOLO format with train/val/test splits).

Expected raw layout after download:
```
data/raw/RDD_SPLIT/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

The project creates a new 2-class dataset at:
```
data/rdd2class_yolo/
  images/{train,val,test}
  labels/{train,val,test}
```

Notes:
- The `data/` directory is ignored by git. You must download/prepare data locally.
- Labels are YOLO format: `class x_center y_center width height` (normalized to [0, 1]).

## Repository structure

```
road-damage-detection/
  api/
    main.py
  configs/
    rdd2class.yaml
  datasets/
    inspect_yolo_labels.py
    make_2class_dataset.py
    preview_classes.py
  demo/
    app.py
  src/
    infer_video.py
  data/                 (ignored)
  requirements.txt
  README.md
  .gitignore
```

## Setup (Windows)

1) Create and activate a virtual environment:
```
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

2) Install dependencies:
```
pip install -r requirements.txt
```

3) Install Kaggle CLI (needed only to download data):
```
pip install kaggle
```

4) Configure Kaggle credentials:
- Download `kaggle.json` from your Kaggle account settings.
- Place it at:
  - `C:\Users\<YOU>\.kaggle\kaggle.json`

Do not commit `kaggle.json`.

## Download RDD2022 (Kaggle)

Run the download script (Windows PowerShell):
```
.\scripts\download_rdd2022.ps1
```

Expected output directory:
```
data/raw/RDD_SPLIT
```

## Build the 2-class dataset

1) (Optional) Inspect label class IDs in the raw dataset:
```
python .\datasets\inspect_yolo_labels.py --root data\raw\RDD_SPLIT --allowed 0,1,2,3,4
```

2) Create the 2-class dataset:
```
python .\datasets\make_2class_dataset.py
```

3) Verify the new dataset contains only class IDs 0 and 1:
```
python .\datasets\inspect_yolo_labels.py --root data\rdd2class_yolo --allowed 0,1
```

## Train (YOLOv8)

Smoke test (quick check that everything runs):
```
yolo train model=yolov8n.pt data=configs\rdd2class.yaml epochs=2 imgsz=640 batch=8 device=0 name=smoke_2ep
```

Full training (better results):
```
yolo train model=yolov8n.pt data=configs\rdd2class.yaml epochs=50 imgsz=640 batch=8 device=0 name=baseline_50ep
```

Training artifacts are saved under:
```
runs/detect/<run_name>/
```

## Video inference (MP4 + CSV)

Run video inference using trained weights:
```
python .\src\infer_video.py --model runs\detect\smoke_2ep\weights\best.pt --source data\sample_video.mp4 --conf 0.25 --out_dir outputs
```

Outputs:
- `outputs/<video_name>_pred.mp4`
- `outputs/<video_name>_pred.csv`

## FastAPI inference service

Start the API:
```
uvicorn api.main:APP --host 127.0.0.1 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict_image` returns JSON detections
- `POST /predict_image_annotated` returns an annotated PNG

Test using Postman:
- Method: POST
- URL: `http://127.0.0.1:8000/predict_image`
- Body: form-data
  - key: `file` (type: File)
  - value: select an image

To get an annotated image:
- URL: `http://127.0.0.1:8000/predict_image_annotated?conf=0.05`

## Streamlit demo

Run the demo:
```
streamlit run demo\app.py
```

The demo supports:
- Image upload: shows annotated output and JSON detections
- Video upload: produces annotated video and CSV and allows downloading them

## Notes

- If you see zero detections with a smoke model, reduce `conf` (e.g., 0.05 or 0.01).
- The dataset is class-imbalanced (cracks are much more frequent than potholes). This is expected.
