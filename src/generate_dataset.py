"""
Generates a synthetic but realistically-structured dataset for the
Mental Health Score Prediction project.

Why synthetic: no real dataset was supplied and Kaggle is not reachable
from this environment, so a generator was built instead of fabricating
a "real" source. The generation logic below encodes plausible,
literature-consistent relationships (e.g. poor sleep + high stress +
low social support -> lower wellbeing score) plus realistic noise,
missing values, duplicates and outliers so the downstream EDA/ML
pipeline has genuine signal to find rather than pure randomness.

Target: Mental_Health_Score, continuous 0-100 (higher = better wellbeing).
This is framed as a general WELLBEING score, not a clinical diagnosis.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

# ---- Demographics ----
age = np.random.randint(18, 60, N)
gender = np.random.choice(["Male", "Female", "Other"], N, p=[0.48, 0.48, 0.04])
employment_status = np.random.choice(["Student", "Employed", "Unemployed"], N, p=[0.45, 0.45, 0.10])

# ---- Lifestyle / behavioral features ----
sleep_hours = np.clip(np.random.normal(6.5, 1.5, N), 2, 11)
sleep_quality = np.clip(np.random.normal(6, 2, N), 1, 10)  # self-rated 1-10
screen_time_hours = np.clip(np.random.normal(5.5, 2.2, N), 0, 14)
physical_activity_hours = np.clip(np.random.exponential(2.5, N), 0, 14)  # hrs/week
social_interaction_score = np.clip(np.random.normal(5.5, 2.3, N), 0, 10)  # 0-10 scale
diet_quality = np.clip(np.random.normal(6, 2, N), 1, 10)
caffeine_intake_cups = np.clip(np.random.poisson(2, N), 0, 10)

# ---- Work / academic features (depend on employment status) ----
work_study_hours = np.where(
    employment_status == "Unemployed",
    np.clip(np.random.normal(1, 1, N), 0, 5),
    np.clip(np.random.normal(7.5, 2.5, N), 0, 16),
)
academic_work_pressure = np.clip(np.random.normal(6, 2, N), 1, 10)
financial_stress = np.clip(np.random.normal(5.5, 2.5, N), 1, 10)

# ---- Support / stress ----
family_support_score = np.clip(np.random.normal(6.5, 2.2, N), 0, 10)
stress_level = np.clip(np.random.normal(5.5, 2.2, N), 1, 10)

# ---- Build target with a plausible weighted formula + noise ----
score = (
    50
    + 3.2 * (sleep_quality - 5)
    + 1.6 * (sleep_hours - 6.5)
    - 2.8 * (stress_level - 5)
    + 2.0 * (social_interaction_score - 5)
    + 1.4 * (family_support_score - 5)
    + 1.1 * (diet_quality - 5)
    + 1.3 * (physical_activity_hours.clip(0, 8) - 2.5)
    - 1.5 * (financial_stress - 5)
    - 0.9 * (academic_work_pressure - 5)
    - 0.6 * (screen_time_hours - 5.5)
    - 0.3 * (caffeine_intake_cups - 2)
    + np.random.normal(0, 6, N)  # irreducible noise
)
mental_health_score = np.clip(score, 0, 100)

df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "Employment_Status": employment_status,
    "Sleep_Hours": sleep_hours.round(1),
    "Sleep_Quality": sleep_quality.round(1),
    "Screen_Time_Hours": screen_time_hours.round(1),
    "Physical_Activity_Hours": physical_activity_hours.round(1),
    "Social_Interaction_Score": social_interaction_score.round(1),
    "Diet_Quality": diet_quality.round(1),
    "Caffeine_Intake_Cups": caffeine_intake_cups,
    "Work_Study_Hours": work_study_hours.round(1),
    "Academic_Work_Pressure": academic_work_pressure.round(1),
    "Financial_Stress": financial_stress.round(1),
    "Family_Support_Score": family_support_score.round(1),
    "Stress_Level": stress_level.round(1),
    "Mental_Health_Score": mental_health_score.round(1),
})

# ---- Inject realistic messiness ----
# 1. Missing values (MCAR-ish) in a handful of columns
for col, frac in [("Sleep_Quality", 0.04), ("Diet_Quality", 0.03),
                   ("Family_Support_Score", 0.05), ("Physical_Activity_Hours", 0.02)]:
    idx = np.random.choice(df.index, int(frac * N), replace=False)
    df.loc[idx, col] = np.nan

# 2. Duplicate rows
dup_idx = np.random.choice(df.index, 25, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# 3. A few extreme outliers
out_idx = np.random.choice(df.index, 10, replace=False)
df.loc[out_idx, "Screen_Time_Hours"] = np.random.uniform(20, 30, 10)

df = df.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle
df.to_csv("/home/claude/mental-health-score-prediction/data/dataset.csv", index=False)
print("Saved dataset:", df.shape)
print(df.head())
