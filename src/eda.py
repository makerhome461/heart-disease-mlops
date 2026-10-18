"""Exploratory data analysis: save plots to report/figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA = "data/processed/heart_clean.csv"
OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
sns.set_theme(style="whitegrid")

print(df.shape)
print(df.describe().T)

# 1. Class balance
plt.figure(figsize=(5, 4))
sns.countplot(x="target", data=df)
plt.title("Class balance (0 = no disease, 1 = disease)")
plt.tight_layout()
plt.savefig(OUT / "class_balance.png", dpi=150)
plt.close()

# 2. Histograms of numeric features, split by target
numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, numeric):
    sns.histplot(data=df, x=col, hue="target", bins=20, ax=ax)
    ax.set_title(col)
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig(OUT / "histograms.png", dpi=150)
plt.close()

# 3. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=150)
plt.close()

print("Saved plots to", OUT)
