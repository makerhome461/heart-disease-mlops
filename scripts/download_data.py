"""Download the Cleveland Heart Disease dataset from UCI."""
import urllib.request
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
OUT_FILE = Path("data/raw/heart_cleveland_raw.csv")


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as response:
        text = response.read().decode("utf-8")
    with open(OUT_FILE, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        f.write(text)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
