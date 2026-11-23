import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from src.prepare import SEED
from src.validate_data import CLASSES


def train_and_score(training, testing, directory):
    model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
    model.fit(*training)
    features, labels = testing
    predictions = model.predict(features)
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "correctas": int(sum(predictions == labels)),
        "n_train": len(training[1]), "n_test": len(labels),
        "clases": CLASSES,
        "matriz_confusion": confusion_matrix(labels, predictions, labels=CLASSES).tolist()
    }
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (directory / "model.pkl").write_bytes(pickle.dumps(model))
    return metrics
