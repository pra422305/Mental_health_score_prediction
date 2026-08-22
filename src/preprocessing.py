"""
Preprocessing pipeline for the Mental Health Score Prediction project.

Kept as a standalone module so app.py, train.py, and the notebook can
all import the exact same logic - one source of truth, no drift
between training-time and inference-time preprocessing.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Mental_Health_Score"

NUMERIC_FEATURES = [
    "Age", "Sleep_Hours", "Sleep_Quality", "Screen_Time_Hours",
    "Physical_Activity_Hours", "Social_Interaction_Score", "Diet_Quality",
    "Caffeine_Intake_Cups", "Work_Study_Hours", "Academic_Work_Pressure",
    "Financial_Stress", "Family_Support_Score", "Stress_Level",
    # engineered features appended in engineer_features()
    "Lifestyle_Score", "Sleep_Efficiency_Score", "Wellbeing_Support_Score",
]

CATEGORICAL_FEATURES = ["Gender", "Employment_Status"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a small number of engineered features, each logically derived
    from existing columns (not invented from nothing):

    - Lifestyle_Score: average of diet, physical activity (capped at 8h
      before scaling to a 0-10-ish range) and (10 - screen time capped at 10)
      -> a single number summarizing healthy daily habits.
    - Sleep_Efficiency_Score: combines sleep hours (closeness to the
      recommended ~8h) with self-rated sleep quality.
    - Wellbeing_Support_Score: average of social interaction and family
      support - a proxy for the person's support network strength.
    """
    df = df.copy()

    activity_capped = df["Physical_Activity_Hours"].clip(upper=8)
    screen_penalty = (10 - df["Screen_Time_Hours"].clip(upper=10))
    df["Lifestyle_Score"] = (
        df["Diet_Quality"] + activity_capped.clip(upper=10) + screen_penalty
    ) / 3

    sleep_closeness = 10 - (df["Sleep_Hours"] - 8).abs() * 1.5
    sleep_closeness = sleep_closeness.clip(lower=0, upper=10)
    df["Sleep_Efficiency_Score"] = (sleep_closeness + df["Sleep_Quality"]) / 2

    df["Wellbeing_Support_Score"] = (
        df["Social_Interaction_Score"] + df["Family_Support_Score"]
    ) / 2

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def load_clean_data(path: str) -> pd.DataFrame:
    """Load raw CSV, drop exact duplicate rows, engineer features."""
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    df = engineer_features(df)
    return df
