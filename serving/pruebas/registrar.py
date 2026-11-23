"""Registro independiente para el runner público, usando el conjunto Iris de sklearn."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from src.prepare import SEED, prepare
from src.train import train_and_score
from src.validate_data import read_csv

contract = Path("data_contract.json")
mlflow.set_experiment("Iris")
with TemporaryDirectory() as temp:
    folder = Path(temp)
    prepare(folder)
    train, test = [read_csv(folder / s) for s in ("train.csv", "test.csv")]
    with mlflow.start_run(run_name="Iris-Compose") as run:
        result = train_and_score(train, test, folder)
        assert result["accuracy"] >= 0.80
        mlflow.log_params({"seed": SEED, "n_estimators":100,"test_size":0.2,"accuracy_min":0.8})
        mlflow.log_metric("accuracy", result["accuracy"])
        for path in [contract, folder / "train.csv", folder / "test.csv"]:
            mlflow.set_tag(path.name + "_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
        mlflow.log_artifacts(str(folder))
        digest = hashlib.sha256((folder/"model.pkl").read_bytes()).hexdigest()
        run_id = run.info.run_id
    Path("approved.json").write_text(json.dumps({run_id:digest}))
    Path(".env").write_text(f"RUN_ID={run_id}\n")
    print("RUN_ID",run_id)
