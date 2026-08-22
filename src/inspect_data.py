"""Step 1: Dataset inspection."""
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

df = pd.read_csv("/home/claude/mental-health-score-prediction/data/dataset.csv")

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)
print("\nDTYPES:\n", df.dtypes)

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()
target = "Mental_Health_Score"
num_cols.remove(target)

print("\nNUMERICAL COLUMNS:", num_cols)
print("CATEGORICAL COLUMNS:", cat_cols)
print("TARGET:", target)

print("\nMISSING VALUES:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nDUPLICATE ROWS:", df.duplicated().sum())

print("\nUNIQUE VALUES (categorical):")
for c in cat_cols:
    print(f"  {c}: {df[c].unique().tolist()}")

print("\nTARGET DISTRIBUTION:\n", df[target].describe())

print("\nCORRELATION WITH TARGET:")
print(df[num_cols + [target]].corr()[target].sort_values(ascending=False))

# Outlier check via IQR
print("\nOUTLIER COUNTS (IQR method):")
for c in num_cols:
    q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[c] < lo) | (df[c] > hi)).sum()
    if n_out > 0:
        print(f"  {c}: {n_out} outliers")

# Potential leakage check: any feature with near-1 correlation to target
leak = df[num_cols + [target]].corr()[target].drop(target)
print("\nPOTENTIAL LEAKAGE (|corr| > 0.9):", leak[leak.abs() > 0.9].to_dict())

# Highly correlated feature pairs
print("\nHIGHLY CORRELATED FEATURE PAIRS (|corr| > 0.7):")
corr_matrix = df[num_cols].corr().abs()
pairs = corr_matrix.where(~pd.np.eye(len(corr_matrix), dtype=bool)) if hasattr(pd, "np") else None
import numpy as np
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
high_corr = corr_matrix.where(mask).stack()
high_corr = high_corr[high_corr > 0.7]
print(high_corr if len(high_corr) else "  None found")
