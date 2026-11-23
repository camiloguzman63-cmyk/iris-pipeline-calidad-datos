import hashlib
import logging
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp

import mlflow
from src.prepare import SEED, prepare
from src.train import train_and_score
from src.validate_data import ROOT, read_csv


def main():
    output = ROOT / "runs"
    output.mkdir(exist_ok=True)
    run = Path(mkdtemp(prefix="iris-", dir=output))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(run / "pipeline.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    try:
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("Iris")
        with mlflow.start_run(run_name=run.name):
            logging.info("1. Preparación en %s", run)
            prepare(run)
            logging.info("2. Validación del contrato")
            training, testing = [read_csv(run / name) for name in ("train.csv", "test.csv")]
            logging.info("Contrato aprobado en ambas particiones")
            logging.info("3. Entrenamiento y evaluación")
            metrics = train_and_score(training, testing, run)
            mlflow.log_params({"seed": SEED, "n_estimators": 100, "test_size": 0.2,
                               "accuracy_min": 0.80})
            for path in [ROOT / "data_contract.json", run / "train.csv", run / "test.csv"]:
                mlflow.set_tag(path.name + "_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
            mlflow.log_metric("accuracy", metrics["accuracy"])
            logging.info("4. Pruebas de aceptación")
            result = subprocess.run([sys.executable, "tests/test_model.py", str(run)],
                                    cwd=ROOT, capture_output=True, text=True)
            logging.info("%s", result.stderr.strip())
            result.check_returncode()
            logging.info("Modelo aprobado. Accuracy %.4f", metrics["accuracy"])
            mlflow.log_artifacts(str(run))
        return 0
    except Exception:
        logging.exception("Pipeline detenido")
        return 1


if __name__ == "__main__":
    sys.exit(main())
