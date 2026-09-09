🚚 Delivery Demand Forecasting — End-to-End MLOps Pipeline

Hourly hub-level order demand prediction for a Brazilian delivery marketplace. Built as a production-grade MLOps system covering the full lifecycle — from raw data to deployed API.

[![Live API](https://img.shields.io/badge/Live_API-Render-blue?style=for-the-badge&logo=render)](https://demand-forecasting-time-series-1.onrender.com/predict_form)
[![MLflow](https://img.shields.io/badge/Experiments-DagsHub_MLflow-orange?style=for-the-badge&logo=mlflow)](https://dagshub.com/manismansuri24/Demand-Forecasting-Time-Series.mlflow/)
[![Docker](https://img.shields.io/badge/Docker-GHCR-2496ED?style=for-the-badge&logo=docker)](https://ghcr.io/mo-anis-24/delivery-forecast)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/Mo-Anis-24/Demand-Forecasting-Time-Series/ci.yml?style=for-the-badge&label=CI/CD)](https://github.com/Mo-Anis-24/Demand-Forecasting-Time-Series/actions)

---
📌 Problem Statement

A Brazilian delivery marketplace operates across **26 hubs** in 4 cities. Each hub needs to know **how many orders to expect in the next hour** to allocate the right number of drivers. Too few drivers = late deliveries. Too many = wasted costs.

Goal: Predict hourly order count per hub using historical order data (370K+ orders across Q1 2021).

---

## 🏆 Key Results

| Metric | Seasonal Naive Baseline | LightGBM (Tuned) | Improvement |
|--------|------------------------|-------------------|-------------|
| MAE    | 2.22                   | **1.64**          | **26.1%**   |
| WAPE   | 43.4%                  | **32.1%**         | **26.0%**   |
| RMSE   | 4.35                   | **3.20**          | **26.4%**   |

- **30 Optuna trials** for hyperparameter search
- **35 tracked experiments** on DagsHub (publicly viewable)
- Feature engineering drove 26% improvement; hyperparameter tuning added ~1% more

---

## 🏗️ Architecture

```
Raw CSVs (7 files, 370K orders)
    │
    ▼
┌─────────────────────┐
│   Data Ingestion     │  Join 7 tables → clean → save parquet
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Data Transformation  │  Hub-hour aggregation → 33 features
└─────────────────────┘   (lags, rolling stats, cyclical, holidays)
    │
    ▼
┌─────────────────────┐
│   Model Trainer      │  LightGBM (Poisson) + MLflow logging
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Hyperparameter Tuner │  Optuna (30 trials) + MLflow nested runs
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   FastAPI Serving    │  /predict (CSV) + /predict_form (web form)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Docker + CI/CD      │  GitHub Actions → GHCR → Render
└─────────────────────┘
```

---

## 🔗 Live Links

| What | Link |
|------|------|
| **Live Prediction Form** | [Try it →](https://demand-forecasting-time-series-1.onrender.com/predict_form) |
| **Swagger API Docs** | [API Docs →](https://demand-forecasting-time-series-1.onrender.com/docs) |
| **MLflow Experiment Dashboard** | [35 Runs on DagsHub →](https://dagshub.com/manismansuri24/Demand-Forecasting-Time-Series.mlflow/) |
| **Docker Image** | `ghcr.io/mo-anis-24/delivery-forecast:latest` |

> **Note:** The Render free tier sleeps after 15 min of inactivity. First request may take ~30 seconds to wake up.

---

## 📊 Dataset

**Source:** Brazilian Delivery Center (Kaggle-style, Q1 2021)

| Table | Rows | Description |
|-------|------|-------------|
| Orders | 368,999 | Order timestamps, status, hub assignment |
| Stores | 951 | Store metadata, segment |
| Hubs | 32 | Hub location, city, state |
| Drivers | 930 | Driver info |
| Deliveries | 352,020 | Delivery details |
| Payments | 368,999 | Payment info |
| Channels | 9 | Order channels |

**After processing:** 66,092 rows × 33 features across 26 active hubs (dropped 6 low-activity hubs).

---

## 🧠 Feature Engineering (33 Features)

The biggest improvement came from careful feature engineering, not model selection.

**Calendar Features:**
- `hour`, `dow`, `day`, `month`, `week`, `is_weekend`

**Cyclical Encodings:**
- `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` — captures circular nature of time

**Activity Windows:**
- `is_active_window` (13:00–00:00), `is_dead_window` (02:00–12:00)
- `is_evening_peak` (18:00–22:00), `is_late_night` (22:00–00:00)

**Special Days:**
- `is_holiday` (Brazilian holidays), `is_payday_window` (days 5-6, 20-21)

**Lag Features:**
- `lag_1`, `lag_24`, `lag_168` (1 week), `lag_336` (2 weeks)

**Rolling Statistics:**
- `roll_mean_24`, `roll_mean_168`, `roll_std_24`, `roll_std_168`, `roll_max_24`

**Hub-Level:**
- `hub_id`, `hub_city`, `hub_state`, `hub_dominant_segment`, `hub_dominant_channel`
- `hub_avg_hourly`, `same_hour_2w_mean`

**Top predictors by importance:** `hub_id`, `lag_1`, `roll_mean_24`, `lag_24`, `roll_std_24`

---

## 🔬 Model Details

**Algorithm:** LightGBM with Poisson objective (natural fit for count data)

**Best Hyperparameters (from Optuna):**

| Parameter | Value |
|-----------|-------|
| learning_rate | 0.033 |
| num_leaves | 130 |
| max_depth | 6 |
| min_child_samples | 92 |
| colsample_bytree | 0.996 |
| reg_alpha | 0.011 |
| reg_lambda | 0.015 |

**Training Strategy:**
- Time-based split: last 14 days = test set
- Validation: 7 days before test (for Optuna)
- Early stopping: 100 rounds (final), 50 rounds (trials)
- Leakage validation: lag integrity checked at 100%

---

## 📈 Experiment Tracking

All experiments are tracked with **MLflow** and hosted on **DagsHub** (publicly viewable).

**[→ View all 35 experiments on DagsHub](https://dagshub.com/manismansuri24/Demand-Forecasting-Time-Series.mlflow/)**

Each run logs:
- All hyperparameters
- Test metrics (MAE, RMSE, WAPE, improvement percentages)
- Trained model artifact (downloadable)
- Feature importance CSV
- Test predictions CSV

Run structure:
```
delivery-hourly-forecast (experiment)
├── lgbm_default          — baseline model (MAE: 1.64)
└── optuna_tuning         — hyperparameter search
    ├── trial_0           — val_mae: 2.15
    ├── trial_1           — val_mae: 1.98
    ├── ...
    └── trial_29          — val_mae: 1.87
```

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Redirects to Swagger docs |
| `GET` | `/predict_form` | Interactive prediction form (web UI) |
| `POST` | `/predict_form` | Submit form → get prediction |
| `POST` | `/predict` | Upload CSV → get predictions table |
| `GET` | `/train` | Retrain model (model_trainer + hyperparameter_tuning) |

### Web Form Prediction

Visit `/predict_form` to:
1. Select a delivery hub from dropdown (26 hubs with auto-filled city/state)
2. Set the hour, day, month
3. Enter recent order history
4. Click "Predict Demand" → see predicted order count

---

## 📁 Project Structure

```
├── app.py                          # FastAPI application
├── Dockerfile                      # Docker containerization
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── .github/workflows/ci.yml        # CI/CD pipeline
│
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Utilities + MLflow setup
│   ├── components/
│   │   ├── data_ingestion.py       # Raw data loading + cleaning
│   │   ├── data_transformation.py  # Feature engineering
│   │   ├── model_trainer.py        # Default LightGBM + MLflow
│   │   └── hyperparameter_tuning.py # Optuna search + MLflow
│   ├── pipeline/
│   │   └── train_pipeline.py       # Orchestrates all 4 stages
│   ├── exception/
│   │   └── exception.py            # Custom exception handler
│   └── logger/
│       └── logger.py               # Custom logging
│
├── final_model/
│   ├── model.pkl                   # Trained LightGBM model
│   └── transformation.pkl          # Feature schema + category mappings
│
├── templates/
│   ├── form.html                   # Prediction input form
│   ├── result.html                 # Prediction result page
│   └── table.html                  # CSV prediction results
│
├── notebook/
│   ├── 01_eda_new.ipynb            # Exploratory data analysis
│   ├── 02_feture_eng_new.ipynb     # Feature engineering exploration
│   ├── 03_model.ipynb              # Model development
│   └── 04_tuning.ipynb             # Hyperparameter tuning
│
└── artifacts/                      # Generated during pipeline run
    ├── data_ingestion/
    ├── data_transformation/
    ├── model_trainer/
    └── hyperparameter_tuning/
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.12
- conda (miniconda or anaconda)

### Local Setup

```bash
# Clone the repo
git clone https://github.com/Mo-Anis-24/Demand-Forecasting-Time-Series.git
cd Demand-Forecasting-Time-Series

# Create conda environment
conda create -n delivery-forecast python=3.12 -y
conda activate delivery-forecast

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the API
python app.py
# Open: http://localhost:8000/predict_form
```

### Run with Docker

```bash
# Pull the image
docker pull ghcr.io/mo-anis-24/delivery-forecast:latest

# Run
docker run -p 8000:8000 ghcr.io/mo-anis-24/delivery-forecast:latest

# Open: http://localhost:8000/predict_form
```

### Run Training Pipeline

Requires raw CSV files in `data/raw/` (download from the dataset source).

```bash
python -m src.pipeline.train_pipeline
```

---

## 🔄 CI/CD Pipeline

Every `git push` to `main` triggers:

```
Push to GitHub
    │
    ▼
GitHub Actions
    ├── Job 1: Test
    │   ├── Install dependencies
    │   ├── Test all imports
    │   └── Verify model files exist
    │
    └── Job 2: Build & Push (if tests pass)
        ├── Build Docker image
        └── Push to GHCR (ghcr.io/mo-anis-24/delivery-forecast)
            │
            ▼
        Render auto-deploys from GitHub
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.12 |
| **ML Model** | LightGBM (Poisson objective) |
| **Hyperparameter Tuning** | Optuna (TPE sampler, 30 trials) |
| **Experiment Tracking** | MLflow on DagsHub |
| **Feature Engineering** | pandas, numpy, holidays |
| **API Framework** | FastAPI + Uvicorn |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Docker Registry** | GitHub Container Registry (GHCR) |
| **Cloud Deployment** | Render |
| **Version Control** | Git + GitHub |

---

## 📝 Key Learnings

1. **Feature engineering > model tuning:** Careful lag and rolling features gave 26% improvement. Optuna's 30-trial search added only ~1% more. For tabular time series, domain-specific features matter most.

2. **Categorical encoding at prediction time is tricky:** LightGBM stores category codes during training. At prediction time, you must apply the exact same encoding — saving `cat_mappings` in `transformation.pkl` solved this.

3. **MLflow runs need explicit `end_run()`:** When using DagsHub as the MLflow backend, runs can get stuck in "RUNNING" state. Adding `mlflow.end_run(status="FINISHED")` explicitly fixes this.

4. **Docker needs system libraries:** LightGBM requires `libgomp1` (OpenMP) which isn't included in `python:3.12-slim`. One line in the Dockerfile fixes it: `apt-get install -y libgomp1`.

5. **Folder names with spaces break everything:** Learned this the hard way. Always use hyphens or underscores in project folder names.

---

## 👤 Author

**Anish (Mo-Anis-24)**

- GitHub: [Mo-Anis-24](https://github.com/Mo-Anis-24)
- DagsHub: [manismansuri24](https://dagshub.com/manismansuri24)

---

## 🙏 Acknowledgments

- **Krish Naik** — Modular coding architecture and MLOps pipeline design
- **Brazilian Delivery Center Dataset** — Source data
- **DagsHub** — Free MLflow hosting for experiment tracking
- **Render** — Free cloud deployment

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
