FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py .
COPY src/ src/
COPY app.py .
COPY templates/ templates/
COPY final_model/ final_model/
COPY data_for_docker/features.parquet artifacts/data_transformation/features.parquet

RUN pip install -e .

RUN mkdir -p prediction_output

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]