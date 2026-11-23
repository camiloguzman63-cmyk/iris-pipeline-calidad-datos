"""Uso: python predict.py runs/iris-IDENTIFICADOR entrada.json"""
import argparse
import json
import pickle
from pathlib import Path
from src.validate_data import validate_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("entrada", type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.entrada.read_text(encoding="utf-8"))
        features = validate_input(payload)
        model = pickle.loads((args.run / "model.pkl").read_bytes())
        print(json.dumps({"species": model.predict([features])[0]}))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
