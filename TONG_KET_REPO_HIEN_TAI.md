# TONG KET HIEN TRANG REPO VSL

Ngay cap nhat: 2026-04-16

## 1) Tong quan nhanh

Repo hien tai da co day du he thong nhan dang ngon ngu ky hieu Viet Nam theo huong:
- Huan luyen model (PyTorch + MediaPipe + feature engineering)
- Suy luan realtime/continuous qua backend FastAPI
- Frontend web React + Vite
- Mobile mini app Expo React Native
- Bo benchmark cho continuous va sentence-level
- Bo tai lieu bao cao, quick start, reproducibility, colab guide

Chi so tong quan:
- So file dang duoc theo doi boi git: 5322
- Thu muc cap cao chinh: .venv, backend, baocao, benchmark, custom_videos, frontend, mobile-mini, Videos

## 2) Snapshot Git hien tai

- Nhanh dang dung: V2 (tracking origin/V2)
- Trang thai working tree: co thay doi chua commit trong `backend/train_experimental_comparison.py`
- 5 commit gan nhat:
  1. 74aa6d1 - new
  2. 90c7333 - Add three-phase benchmark workflow
  3. e7da177 - Apply Option 1 profile in Cell 3
  4. dd70b08 - Make download cell resilient to zip failures
  5. e617df3 - new fix

## 3) Cau truc va chuc nang theo module

### 3.1 Root level

- `start.bat`: khoi dong backend + frontend
- `stop.bat`: dung he thong
- `train.bat`: train nhanh
- `consolidate_data.py`: script tong hop du lieu
- `Data.xlsx`: mapping video -> nhan
- `Videos/`: kho video goc
- `custom_videos/`: video bo sung/tu quay
- `README.md`: tai lieu tong quan + huong dan su dung

### 3.2 Backend (`backend/`)

Thanh phan chinh:
- API/inference:
  - `app.py`
  - `demo_camera.py`
  - `benchmark_ondevice.py`
- Huan luyen/chuan hoa:
  - `train_gpu.py`
  - `train_gpu_enhanced.py`
  - `train_baselines.py`
  - `train_ablation_variants.py`
  - `train_phase1_reduced.py`
  - `phase2_fewshot_embedding.py`
  - `train_experimental_comparison.py` (ban framework few-shot/metric-learning moi)
- Trich xuat/quan sat keypoint:
  - `extract_landmarks_only.py`
  - `keypoint_variants.py`
  - `spatial_augmentation.py`
  - `visualize_spatial_aug.py`
  - cac script export keypoint/video preview
- Kiem thu:
  - `test_baseline_models.py`
  - `test_model_load.py`
- Tai lieu backend:
  - `DEBUGGING_INFERENCE.md`
  - `KEYPOINT_USAGE.md`
  - `CONFIDENCE_THRESHOLDS.md`
  - `TTA_TEMPORAL_SMOOTHING.md`

Dependency backend (`backend/requirements.txt`):
- fastapi, uvicorn, websockets
- mediapipe, opencv-python-headless
- numpy, openpyxl, scikit-learn, python-multipart

Model artifact da co:
- `backend/models/`: labels, norm stats, model checkpoints, training history
- `backend/models_15cls_run1/`: labels, norm stats, model checkpoints, training history

Du lieu trung gian da co:
- `backend/landmarks/`
- `backend/landmarks_extract_only/`
- `backend/keypoint_previews/`

### 3.3 Benchmark (`benchmark/`)

Cau truc:
- `benchmark/continuous/`
  - scripts: detect active span, build dataset, generate predictions tu backend, evaluate CER/WER, plot metrics
  - data da co: continuous_dataset.real.json, continuous_predictions.real.json, continuous_eval.real.json, report markdown, rendered clips
- `benchmark/sentence_level/`
  - scripts: validate dataset, split dataset, generate dataset tu Data.xlsx, generate predictions, evaluate sentence benchmark, build full manifest, n-gram LM
  - data da co: sample/synthetic datasets, split reports, prediction files, evaluation files
- `benchmark/scripts/`
  - `phase3_continuous_sentence_audit.py`
  - script tong hop/analyze
- `benchmark/report_assets/`
  - charts + so lieu snapshot cho bao cao

### 3.4 Frontend web (`frontend/`)

Cong nghe:
- React 19 + Vite 6 + react-router-dom + i18next

Cau truc:
- `index.html`, `vite.config.js`, `package.json`
- `src/App.jsx`, `src/main.jsx`, `src/i18n.js`, `src/styles.css`
- Pages:
  - `src/pages/CameraPage.jsx`
  - `src/pages/ContinuousPage.jsx`
  - `src/pages/UploadPage.jsx`
  - `src/pages/TrainingPage.jsx`
  - `src/pages/StatusPage.jsx`

### 3.5 Mobile mini (`mobile-mini/`)

Cong nghe:
- Expo 54 + React Native 0.81 + expo-camera

File chinh:
- `App.js`, `index.js`, `app.json`, `eas.json`
- script nhanh: `start-expo-go.bat`, `build-apk.bat`
- huong dan: `mobile-mini/README.md`

### 3.6 Tai lieu va bao cao (`baocao/`)

Da co bo tai lieu day du cho:
- Tong quan phan tich du an
- Lo trinh benchmark
- Reproducibility
- Colab quick start/pro guide
- Three-phase execution
- Nhiem vu da hoan thanh (task completion)
- Mau bao cao NCKH

## 4) Luong van hanh da san sang

Luong co ban:
1. Chuan bi du lieu (Data.xlsx + Videos)
2. Extract/cached landmarks
3. Train model
4. Chay backend inference
5. Chay benchmark continuous/sentence
6. Tong hop metrics/report

Lenh/vien script mau:
- Khoi dong he thong: `start.bat`
- Dung he thong: `stop.bat`
- Train nhanh: `train.bat`
- Continuous pipeline nhanh: `benchmark/continuous/run_pipeline_real.bat`
- Sentence pipeline nhanh: `benchmark/sentence_level/run_pipeline.bat`

## 5) Diem manh hien tai cua repo

- Da co day du stack end-to-end (train -> deploy backend -> web/mobile -> benchmark)
- Da co benchmark scaffold cho bai toan chuoi (continuous + sentence)
- Da co tai lieu van hanh va reproducibility kha day du
- Da co huong three-phase benchmark va script audit bo sung
- Da co huong nghien cuu few-shot/embedding trong backend

## 6) Trang thai dang chu y

- Working tree hien tai con 1 file thay doi chua commit: `backend/train_experimental_comparison.py`
- Co ca `README` (rong/whitespace) va `README.md`; nen duyet de tranh trung/noise
- Co su hien dien cua thu muc lon nhu `.venv`, `node_modules`, `landmarks`, `Videos` (can tiep tuc quan ly bang .gitignore/hinh thuc luu tru phu hop)

## 7) De xuat tiep theo (neu can)

- Chot va commit file `backend/train_experimental_comparison.py` neu da on dinh
- Tao them 1 file MUC_LUC_REPO.md theo kieu index click-through neu can tra cuu nhanh
- Dong bo hoa huong dan README.md voi huong three-phase moi nhat
- Tao script auto-snapshot (tu dong cap nhat tong ket hien trang theo branch + commit + so lieu)

---

Tai lieu nay dong vai tro snapshot hien trang tong hop de phuc vu quan ly du an va bao cao tien do.
