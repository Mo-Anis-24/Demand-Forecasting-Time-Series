import sys
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.exception.exception import forcast
from src.logger import logger
from src.utils import load_object

from fastapi import FastAPI, File, UploadFile, Request
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


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    return Response(
        "Training requires raw CSVs in data/raw/. "
        "Run: python -m src.pipeline.train_pipeline"
    )


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        logger.logging.info(f"Received prediction request — shape: {df.shape}")

        transformation = load_object("final_model/transformation.pkl")
        final_model    = load_object("final_model/model.pkl")

        feature_cols = transformation["feature_cols"]
        cat_features = transformation["cat_features"]
        cat_mappings = transformation["cat_mappings"]

        # Convert text columns to numbers using saved mappings
        # "SÃO PAULO" → 0, "RIO DE JANEIRO" → 1, etc.
        for col in cat_features:
            if col in df.columns and col in cat_mappings:
                categories = cat_mappings[col]
                mapping = {val: idx for idx, val in enumerate(categories)}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)

        X = df[feature_cols]

        # Predict using numpy array
        y_pred = final_model.booster_.predict(X.values).clip(0)

        df["predicted_orders"] = y_pred.round(2)
        logger.logging.info(f"Predictions generated — mean: {y_pred.mean():.2f}")

        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse(
            "table.html",
            {"request": request, "table": table_html},
        )

    except Exception as e:
        raise forcast(e, sys)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)