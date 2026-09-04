import sys
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

from src.exception.exception import forcast
from src.logger import logger
from src.utils import load_object

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run


app = FastAPI(title="Delivery Demand Forecast API")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")

transformation = load_object("final_model/transformation.pkl")
final_model    = load_object("final_model/model.pkl")

hub_lookup = {}
for hub in transformation["hub_info"]:
    hub_lookup[hub["hub_id"]] = {
        "hub_name": hub.get("hub_name", ""),
        "hub_city": hub.get("hub_city", ""),
        "hub_state": hub.get("hub_state", ""),
    }

seg_map = transformation.get("hub_dominant_segment", {})
ch_map  = transformation.get("hub_dominant_channel", {})
for hid in hub_lookup:
    hub_lookup[hid]["hub_dominant_segment"] = seg_map.get(hid, "GOOD")
    hub_lookup[hid]["hub_dominant_channel"] = ch_map.get(hid, "FOOD")


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        from src.components.model_trainer import ModelTrainer
        from src.components.hyperparameter_tuning import HyperparameterTuner

        trainer = ModelTrainer()
        trainer_metrics = trainer.initiate_model_trainer()

        tuner = HyperparameterTuner()
        tuner_metrics = tuner.initiate_hyperparameter_tuning()

        return Response(
            f"Training complete. "
            f"Default MAE: {trainer_metrics['lgbm_mae']}, "
            f"Tuned MAE: {tuner_metrics['tuned_lgbm_mae']}, "
            f"Improvement: {tuner_metrics['total_improvement_mae']}%"
        )
    except Exception as e:
        raise forcast(e, sys)


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        feature_cols = transformation["feature_cols"]
        cat_features = transformation["cat_features"]
        cat_mappings = transformation["cat_mappings"]

        for col in cat_features:
            if col in df.columns and col in cat_mappings:
                categories = cat_mappings[col]
                mapping = {val: idx for idx, val in enumerate(categories)}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)

        X = df[feature_cols]
        y_pred = final_model.booster_.predict(X.values).clip(0)
        df["predicted_orders"] = y_pred.round(2)

        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        raise forcast(e, sys)


@app.get("/predict_form")
async def show_predict_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "hubs": hub_lookup})


@app.post("/predict_form")
async def predict_form_route(
    request: Request,
    hub_id: int = Form(...),
    hour: int = Form(...),
    dow: int = Form(...),
    day: int = Form(...),
    month: int = Form(...),
    is_weekend: int = Form(...),
    lag_1: float = Form(...),
    lag_24: float = Form(...),
    lag_168: float = Form(...),
    lag_336: float = Form(...),
    roll_mean_24: float = Form(...),
    roll_mean_168: float = Form(...),
):
    try:
        hub = hub_lookup.get(hub_id, {})
        hub_city = hub.get("hub_city", "SAO PAULO")
        hub_state = hub.get("hub_state", "SP")
        hub_dominant_segment = hub.get("hub_dominant_segment", "GOOD")
        hub_dominant_channel = hub.get("hub_dominant_channel", "FOOD")
        hub_name = hub.get("hub_name", "Unknown")

        week = pd.Timestamp(year=2021, month=month, day=day).isocalendar().week
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        is_active_window = int((hour >= 13 and hour <= 23) or hour == 0)
        is_dead_window = int(2 <= hour <= 12)
        is_evening_peak = int(18 <= hour <= 22)
        is_late_night = int(hour >= 22 or hour == 0)
        is_holiday = 0
        is_payday_window = int(day in [5, 6, 20, 21])
        roll_std_24 = abs(lag_1 - lag_24) * 0.5
        roll_std_168 = abs(lag_168 - lag_336) * 0.5
        roll_max_24 = max(lag_1, lag_24)
        same_hour_2w_mean = (lag_168 + lag_336) / 2
        hub_avg_hourly = roll_mean_168

        data = {
            "hub_id": hub_id,
            "hub_city": hub_city,
            "hub_state": hub_state,
            "hub_dominant_segment": hub_dominant_segment,
            "hub_dominant_channel": hub_dominant_channel,
            "hour": hour, "dow": dow, "day": day, "month": month,
            "week": int(week), "is_weekend": is_weekend,
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "dow_sin": dow_sin, "dow_cos": dow_cos,
            "is_active_window": is_active_window,
            "is_dead_window": is_dead_window,
            "is_evening_peak": is_evening_peak,
            "is_late_night": is_late_night,
            "is_holiday": is_holiday,
            "is_payday_window": is_payday_window,
            "lag_1": lag_1, "lag_24": lag_24,
            "lag_168": lag_168, "lag_336": lag_336,
            "roll_mean_24": roll_mean_24,
            "roll_mean_168": roll_mean_168,
            "roll_std_24": roll_std_24,
            "roll_std_168": roll_std_168,
            "roll_max_24": roll_max_24,
            "same_hour_2w_mean": same_hour_2w_mean,
            "hub_avg_hourly": hub_avg_hourly,
        }

        df = pd.DataFrame([data])

        feature_cols = transformation["feature_cols"]
        cat_features = transformation["cat_features"]
        cat_mappings = transformation["cat_mappings"]

        for col in cat_features:
            if col in df.columns and col in cat_mappings:
                categories = cat_mappings[col]
                mapping = {val: idx for idx, val in enumerate(categories)}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)

        X = df[feature_cols]
        y_pred = final_model.booster_.predict(X.values).clip(0)
        predicted_orders = round(float(y_pred[0]), 2)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "hub_id": hub_id,
                "hub_name": hub_name,
                "hub_city": hub_city,
                "hour": hour,
                "day": day,
                "month": month,
                "predicted_orders": predicted_orders,
            },
        )

    except Exception as e:
        raise forcast(e, sys)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)