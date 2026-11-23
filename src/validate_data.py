"""Contrato compartido por los CSV de entrenamiento y la inferencia JSON."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "data_contract.json").read_text(encoding="utf-8"))
FIELDS = CONTRACT["required"]
CLASSES = CONTRACT["clases"]


def validate_input(record):
    if not isinstance(record, dict):
        raise ValueError("Se esperaba un objeto JSON")
    issues = []
    for extra in record.keys() - set(FIELDS):
        issues.append(f"{extra}: campo desconocido")
    for name in FIELDS:
        rule = CONTRACT["properties"][name]
        value = record.get(name)
        if name not in record:
            issues.append(f"{name}: falta el campo")
        elif value is None:
            issues.append(f"{name}: valor nulo")
        elif type(value) not in (int, float):
            issues.append(f"{name}: se esperaba un número")
        elif not rule["minimum"] <= value <= rule["maximum"]:
            issues.append(f"{name}: fuera del intervalo [{rule['minimum']}, {rule['maximum']}] cm")
    if issues:
        raise ValueError(" | ".join(issues))
    return [record[name] for name in FIELDS]


def read_csv(path):
    features, labels = [], []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        expected = FIELDS + ["species"]
        if sorted(reader.fieldnames or []) != sorted(expected):
            raise ValueError(f"{path.name}: cabecera incorrecta; se requiere {expected}")
        for row in reader:
            try:
                if None in row:
                    raise ValueError("hay columnas adicionales")
                label = row.pop("species")
                if label not in CLASSES:
                    raise ValueError(f"species: clase desconocida o nula ({label!r})")
                for name, raw in row.items():
                    if raw is None or not raw.strip():
                        row[name] = None
                    else:
                        try:
                            row[name] = float(raw)
                        except ValueError:
                            raise ValueError(f"{name}: se esperaba un número") from None
                features.append(validate_input(row))
                labels.append(label)
            except ValueError as error:
                raise ValueError(f"{path.name}, fila {reader.line_num}: {error}") from error
    if not labels:
        raise ValueError(f"{path.name}: archivo sin observaciones")
    return features, labels
