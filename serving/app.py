"""Serving del modelo Iris aprobado, seleccionado por RUN_ID."""
import hashlib
import json
import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
contract_path = ROOT / "data_contract.json"
contract = json.loads(contract_path.read_text())
fields = contract["required"]
audit = logging.getLogger("iris.predictions")
audit.setLevel(logging.INFO)
audit.addHandler(logging.StreamHandler())
audit.propagate = False


def measurement(name):
    rule = contract["properties"][name]
    return Field(ge=rule["minimum"], le=rule["maximum"], description="Medida en cm")


class Flower(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False,
                              json_schema_extra={"examples": [contract["ejemplos"]["valida"]]})
    sepal_length: float = measurement("sepal_length")
    sepal_width: float = measurement("sepal_width")
    petal_length: float = measurement("petal_length")
    petal_width: float = measurement("petal_width")


class Prediction(BaseModel):
    prediction: Literal["setosa", "versicolor", "virginica"]
    run_id: str
    request_id: str


class ModelInfo(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    training_date: datetime
    run_id: str
    model_sha256: str


@asynccontextmanager
async def lifespan(app):
    run_id = os.environ["RUN_ID"]
    approved = json.loads((ROOT / "approved.json").read_text())
    digest = approved[run_id]
    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise RuntimeError("La ejecución aún no está aprobada")
    if hashlib.sha256(contract_path.read_bytes()).hexdigest() != run.data.tags["data_contract.json_sha256"]:
        raise RuntimeError("El contrato difiere del empleado en entrenamiento")
    with TemporaryDirectory() as directory:
        artifact = Path(client.download_artifacts(run_id, "model.pkl", directory)).read_bytes()
    if hashlib.sha256(artifact).hexdigest() != digest:
        raise RuntimeError("El artefacto difiere del modelo aprobado")
    app.state.model = pickle.loads(artifact)
    app.state.info = ModelInfo(run_id=run_id, accuracy=run.data.metrics["accuracy"],
                               training_date=datetime.fromtimestamp(run.info.start_time / 1000, timezone.utc),
                               model_sha256=digest)
    yield


app = FastAPI(title="Clasificación de flores Iris", version="1.0", lifespan=lifespan,
              description="Cuatro medidas en centímetros para identificar la especie.", docs_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/docs", include_in_schema=False)
def swagger():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Iris | Swagger UI",
                               swagger_js_url="/static/swagger-ui-bundle.js",
                               swagger_css_url="/static/swagger-ui.css")


@app.middleware("http")
async def audit_prediction(request: Request, call_next):
    if request.url.path != "/predict" or request.method != "POST":
        return await call_next(request)
    request.state.id = str(uuid4())
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request.state.id
        return response
    finally:
        audit.info(json.dumps(dict(event="prediction", timestamp=datetime.now(timezone.utc).isoformat(),
                                   request_id=request.state.id, run_id=app.state.info.run_id, status=status,
                                   prediction=getattr(request.state, "species", None),
                                   elapsed_ms=round((time.perf_counter() - started) * 1000, 3))))


@app.get("/health", summary="Consultar disponibilidad")
def health():
    return {"status": "ok", "run_id": app.state.info.run_id}


@app.get("/model-info", summary="Consultar versión y evaluación", response_model=ModelInfo)
def model_info():
    return app.state.info


@app.post("/predict", summary="Clasificar una flor", response_model=Prediction)
def predict(flower: Flower, request: Request):
    vector = [getattr(flower, name) for name in fields]
    species = str(app.state.model.predict([vector])[0])
    request.state.species = species
    return Prediction(prediction=species, run_id=app.state.info.run_id, request_id=request.state.id)
