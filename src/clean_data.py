"""Clean the raw Cleveland data and save a processed copy."""
from pathlib import Path

import pandas as pd

RAW = "data/raw/heart_cleveland_raw.csv"
OUT = "data/processed/heart_clean.csv"

df = pd.read_csv(RAW, na_values="?")
print("Rows before cleaning:", len(df))
print("Missing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])

df = df.dropna().reset_index(drop=True)
df["target"] = (df["target"] > 0).astype(int)

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print("Rows after cleaning:", len(df))
print("Target counts (0 = no disease, 1 = disease):")
print(df["target"].value_counts())
print("Saved", OUT)
