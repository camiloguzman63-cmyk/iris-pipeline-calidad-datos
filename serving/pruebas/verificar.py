"""Pruebas HTTP con curl. Ejecutar con Python 3 desde la carpeta del proyecto."""
import argparse
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--api", default="http://localhost:8000")
parser.add_argument("--out", type=Path, default=Path("evidencias"))
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
contract = json.loads((Path(__file__).resolve().parents[1] / "data_contract.json").read_text())
fields = contract["required"]


def get(path):
    with urlopen(args.api + path, timeout=15) as response:
        return json.load(response)


def predict(payload, expected):
    cmd = ["curl", "--silent", "--show-error", "--max-time", "15", "-H", "Content-Type: application/json",
           "-d", json.dumps(payload), "-w", "\n%{http_code} %{time_total} %{content_type}", args.api + "/predict"]
    body, summary = subprocess.check_output(cmd, text=True).rsplit("\n", 1)
    status, duration, mime = summary.split(maxsplit=2)
    assert int(status) == expected and mime.startswith("application/json"), summary
    return json.loads(body), round(float(duration) * 1000, 2)


info = get("/model-info")
health = get("/health")
assert health["status"] == "ok" and health["run_id"] == info["run_id"]
cases = [([5.1, 3.5, 1.4, .2], "setosa"), ([4.9, 3, 1.4, .2], "setosa"),
         ([6, 2.9, 4.5, 1.5], "versicolor"), ([6.7, 3.1, 4.7, 1.5], "versicolor"),
         ([6.5, 3, 5.8, 2.2], "virginica")]
results = []
for values, expected in cases:
    body, ms = predict(dict(zip(fields, values)), 200)
    assert set(body) == {"prediction", "run_id", "request_id"}
    assert body["prediction"] == expected and body["run_id"] == info["run_id"]
    results.append(dict(input=values, ms=ms, **body))
    print(f"HTTP 200 | {values} | {body['prediction']} | {ms:.2f} ms")
    print(json.dumps(body))
assert len({r["request_id"] for r in results}) == len(cases)
invalid, _ = predict(contract["ejemplos"]["invalida"], 422)
assert len(invalid["detail"]) == 3
print("HTTP 422 | tres errores del contrato")
sample = contract["ejemplos"]["valida"]
for bad in [dict(sample, sepal_length=True), dict(sample, sepal_length="5.1"),
            dict(sample, otro=1), {k: v for k, v in sample.items() if k != "petal_width"}]:
    predict(bad, 422)
(args.out / "resultados.json").write_text(json.dumps(dict(info=info, health=health,
    results=results, invalid=invalid), ensure_ascii=False, indent=2))
