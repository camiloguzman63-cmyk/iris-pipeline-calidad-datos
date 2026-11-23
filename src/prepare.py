import csv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.validate_data import FIELDS

SEED = 42


def prepare(directory):
    iris = load_iris()
    train, test = train_test_split(range(len(iris.target)), test_size=0.2,
                                   stratify=iris.target, random_state=SEED)
    if set(train) & set(test):
        raise ValueError("Hay observaciones compartidas entre las particiones")
    for name, indices in [("train", train), ("test", test)]:
        with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS + ["species"])
            for i in indices:
                writer.writerow([*iris.data[i], iris.target_names[iris.target[i]]])
