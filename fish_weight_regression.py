import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------- 1. Pick & explore ----------
df = pd.read_csv("Fish.csv")
print("Shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSpecies counts:\n", df["Species"].value_counts())
print("\nSummary stats:\n", df.describe())

# Data-quality check: a fish can't weigh 0 grams -> this is a bad row, not a
# real outlier, so we drop it rather than let it distort the fit.
bad_rows = df[df["Weight"] <= 0]
print(f"\nRows with Weight <= 0 (dropped): {len(bad_rows)}")
df = df[df["Weight"] > 0].reset_index(drop=True)
print("Shape after cleaning:", df.shape)

print("\nCorrelation with Weight (numeric feats):\n",
      df.corr(numeric_only=True)["Weight"].sort_values(ascending=False))

# ---------- 2. Prepare & split ----------
# Species is categorical -> one-hot encode it so the model can use it
df_enc = pd.get_dummies(df, columns=["Species"], drop_first=True)

feats = [c for c in df_enc.columns if c != "Weight"]
X, y = df_enc[feats], df_enc["Weight"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)

# ---------- 3. Train ----------
model = LinearRegression().fit(Xtr, ytr)

# ---------- 4. Evaluate (on TEST set only) ----------
pred = model.predict(Xte)
rmse = np.sqrt(mean_squared_error(yte, pred))
r2 = r2_score(yte, pred)

print("\n--- Test set performance ---")
print(f"RMSE: {rmse:,.2f}")
print(f"R2  : {r2:.4f}")

print("\n--- Coefficients ---")
for f, c in zip(feats, model.coef_):
    print(f"{f:>20}: {c:>12,.2f}")
print(f"{'Intercept':>20}: {model.intercept_:>12,.2f}")

# Predicted-vs-actual plot
plt.figure(figsize=(6, 6))
plt.scatter(yte, pred, alpha=0.6, edgecolor="k", linewidth=0.3)
lims = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
plt.plot(lims, lims, "r--", label="perfect prediction")
plt.xlabel("Actual Weight (g)")
plt.ylabel("Predicted Weight (g)")
plt.title(f"Predicted vs Actual (Test set)\nR2={r2:.3f}, RMSE={rmse:,.1f}")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150)
print("\nSaved plot to predicted_vs_actual.png")
