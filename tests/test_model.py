import json
import pickle
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validate_data import CLASSES, CONTRACT, FIELDS, read_csv, validate_input

RUN = Path(sys.argv.pop()) if __name__ == "__main__" else None


class PruebasIris(unittest.TestCase):
    def test_ejemplo_valido(self):
        self.assertEqual(len(validate_input(CONTRACT["ejemplos"]["valida"])), 4)

    def test_ejemplo_invalido(self):
        with self.assertRaises(ValueError):
            validate_input(CONTRACT["ejemplos"]["invalida"])

    def test_tipo(self):
        for value in ["5.1", True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_input(dict(CONTRACT["ejemplos"]["valida"], sepal_length=value))

    def test_nulo(self):
        with self.assertRaisesRegex(ValueError, "nulo"):
            validate_input(dict(CONTRACT["ejemplos"]["valida"], petal_width=None))

    def test_limites(self):
        for name, rule in CONTRACT["properties"].items():
            for bound in ["minimum", "maximum"]:
                validate_input(dict(CONTRACT["ejemplos"]["valida"], **{name:rule[bound]}))
            for value in [rule["minimum"]-0.01, rule["maximum"]+0.01, float("nan")]:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_input(dict(CONTRACT["ejemplos"]["valida"], **{name:value}))

    def test_estructura(self):
        for row in [{}, dict(CONTRACT["ejemplos"]["valida"], species="setosa")]:
            with self.assertRaises(ValueError):
                validate_input(row)

    def test_particiones(self):
        for name, count in [("train", 40), ("test", 10)]:
            _, labels = read_csv(RUN / f"{name}.csv")
            self.assertEqual(Counter(labels), {label:count for label in CLASSES})

    def test_modelo_guardado(self):
        model = pickle.loads((RUN / "model.pkl").read_bytes())
        self.assertEqual(model.n_features_in_, len(FIELDS))
        self.assertIn(model.predict([validate_input(CONTRACT["ejemplos"]["valida"])])[0], CLASSES)

    def test_metricas(self):
        metrics = json.loads((RUN / "metrics.json").read_text())
        self.assertEqual(sum(map(sum, metrics["matriz_confusion"])), 30)
        self.assertEqual(metrics["correctas"]/metrics["n_test"], metrics["accuracy"])

    def test_desempeno(self):
        model = pickle.loads((RUN / "model.pkl").read_bytes())
        x, y = read_csv(RUN / "test.csv")
        self.assertGreaterEqual(model.score(x, y), 0.80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
