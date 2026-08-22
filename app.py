"""
Mental Health Score Prediction & Analysis Dashboard
A Streamlit app for an educational, portfolio-level ML project.

IMPORTANT: This app produces an educational wellbeing-score ESTIMATE
based on lifestyle/behavioral survey-style features. It is NOT a
medical or diagnostic tool.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, engineer_features

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

st.set_page_config(page_title="Mental Health Score Dashboard", layout="wide", page_icon="🧠")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    df_eng = engineer_features(df_clean)
    return df, df_clean, df_eng


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


model = load_model()
df_raw, df_clean, df_eng = load_data()
metrics = load_metrics()

RAW_NUMERIC_COLS = [c for c in NUMERIC_FEATURES if c in df_raw.columns]

DISCLAIMER = (
    "**Disclaimer:** This tool produces an *educational* wellbeing-score estimate "
    "from lifestyle and behavioral inputs on a synthetic/demo dataset. It is **not** "
    "a medical or diagnostic instrument and cannot detect depression, anxiety, or any "
    "clinical condition. If you are struggling, please speak with a qualified mental "
    "health professional."
)

st.sidebar.title("🧠 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Dataset Overview", "Exploratory Data Analysis", "Prediction",
     "Model Performance", "Prediction Explanation", "Insights"],
)

# ============================================================
# HOME
# ============================================================
if page == "Home":
    st.title("🧠 Mental Health Score Prediction & Analysis Dashboard")
    st.markdown(
        "An end-to-end machine learning project that estimates a "
        "**wellbeing score (0-100)** from lifestyle, behavioral, and demographic "
        "survey-style inputs."
    )
    st.warning(DISCLAIMER)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{df_raw.shape[0]:,}")
    col2.metric("Features", f"{df_raw.shape[1] - 1}")
    col3.metric("Best Model", metrics["best_baseline_model"])
    col4.metric("Test R²", f"{metrics['tuned_metrics']['R2']:.3f}")

    st.subheader("Objective")
    st.write(
        "Predict a continuous mental wellbeing score using regression, based on "
        "sleep, stress, social support, lifestyle, and work/academic factors, and "
        "surface which factors are most associated with the score - without making "
        "medical or causal claims."
    )

    st.subheader("ML Workflow")
    st.markdown(
        "1. Dataset inspection & cleaning\n"
        "2. Exploratory Data Analysis\n"
        "3. Feature engineering (Lifestyle, Sleep Efficiency, Wellbeing Support scores)\n"
        "4. Preprocessing pipeline (impute → scale/encode) fit only on training data\n"
        "5. Train & compare 6 regression models\n"
        "6. Hyperparameter tuning via GridSearchCV + 5-fold CV\n"
        "7. Feature-importance explainability\n"
        "8. This Streamlit dashboard, backed by the saved model pipeline"
    )

# ============================================================
# DATASET OVERVIEW
# ============================================================
elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df_raw.shape[0])
    c2.metric("Columns", df_raw.shape[1])
    c3.metric("Missing cells", int(df_raw.isnull().sum().sum()))
    c4.metric("Duplicate rows", int(df_raw.duplicated().sum()))

    st.subheader("Target column")
    st.write(f"**{TARGET}** — continuous score, 0-100 (higher = better self-reported wellbeing).")

    st.subheader("Data preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Missing values by column")
    miss = df_raw.isnull().sum()
    miss = miss[miss > 0]
    if len(miss):
        fig = px.bar(x=miss.index, y=miss.values, labels={"x": "Column", "y": "Missing count"},
                     title="Missing values per column")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing values.")

    st.subheader("Column data types")
    st.dataframe(df_raw.dtypes.astype(str).rename("dtype"), use_container_width=True)

# ============================================================
# EDA
# ============================================================
elif page == "Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")

    st.subheader("Target distribution")
    fig = px.histogram(df_clean, x=TARGET, nbins=40, marginal="box",
                        title="Distribution of Mental Health Score")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numerical feature explorer")
    feat = st.selectbox("Choose a numerical feature", RAW_NUMERIC_COLS)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_clean, x=feat, nbins=30, title=f"Histogram of {feat}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df_clean, y=feat, title=f"Boxplot of {feat}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"{feat} vs {TARGET}")
    fig = px.scatter(df_clean, x=feat, y=TARGET, trendline="ols",
                      title=f"{feat} vs {TARGET}", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = df_clean[RAW_NUMERIC_COLS + [TARGET]].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                     title="Correlation heatmap (numerical features + target)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorical feature distribution")
    cat_feat = st.selectbox("Choose a categorical feature", CATEGORICAL_FEATURES)
    fig = px.box(df_clean, x=cat_feat, y=TARGET, title=f"{TARGET} by {cat_feat}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selected feature pairplot")
    top_feats = [c for c in ["Sleep_Quality", "Stress_Level", "Social_Interaction_Score"] if c in df_clean.columns]
    sample = df_clean[top_feats + [TARGET]].sample(min(400, len(df_clean)), random_state=1)
    fig = px.scatter_matrix(sample, dimensions=top_feats + [TARGET],
                             title="Pairplot: key features vs target (sampled)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PREDICTION
# ============================================================
elif page == "Prediction":
    st.title("🎯 Predict Mental Health Score")
    st.warning(DISCLAIMER)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 80, 25)
            gender = st.selectbox("Gender", df_raw["Gender"].unique())
            employment = st.selectbox("Employment Status", df_raw["Employment_Status"].unique())
            sleep_hours = st.slider("Sleep Hours (per night)", 2.0, 12.0, 7.0, 0.5)
            sleep_quality = st.slider("Sleep Quality (1-10)", 1.0, 10.0, 6.0, 0.5)
        with c2:
            screen_time = st.slider("Screen Time (hrs/day)", 0.0, 16.0, 5.0, 0.5)
            physical_activity = st.slider("Physical Activity (hrs/week)", 0.0, 14.0, 3.0, 0.5)
            social_interaction = st.slider("Social Interaction (0-10)", 0.0, 10.0, 5.0, 0.5)
            diet_quality = st.slider("Diet Quality (1-10)", 1.0, 10.0, 6.0, 0.5)
            caffeine = st.number_input("Caffeine Intake (cups/day)", 0, 10, 2)
        with c3:
            work_study_hours = st.slider("Work/Study Hours per day", 0.0, 16.0, 6.0, 0.5)
            academic_pressure = st.slider("Academic/Work Pressure (1-10)", 1.0, 10.0, 5.0, 0.5)
            financial_stress = st.slider("Financial Stress (1-10)", 1.0, 10.0, 5.0, 0.5)
            family_support = st.slider("Family Support (0-10)", 0.0, 10.0, 6.0, 0.5)
            stress_level = st.slider("Stress Level (1-10)", 1.0, 10.0, 5.0, 0.5)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Age": age, "Gender": gender, "Employment_Status": employment,
            "Sleep_Hours": sleep_hours, "Sleep_Quality": sleep_quality,
            "Screen_Time_Hours": screen_time, "Physical_Activity_Hours": physical_activity,
            "Social_Interaction_Score": social_interaction, "Diet_Quality": diet_quality,
            "Caffeine_Intake_Cups": caffeine, "Work_Study_Hours": work_study_hours,
            "Academic_Work_Pressure": academic_pressure, "Financial_Stress": financial_stress,
            "Family_Support_Score": family_support, "Stress_Level": stress_level,
        }])
        input_eng = engineer_features(input_df)
        pred = model.predict(input_eng)[0]
        pred = float(np.clip(pred, 0, 100))

        st.session_state["last_input"] = input_eng
        st.session_state["last_pred"] = pred

        st.subheader("Result")
        colA, colB = st.columns([1, 2])
        with colA:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                title={"text": "Predicted Wellbeing Score"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "#4C78A8"},
                       "steps": [
                           {"range": [0, 40], "color": "#f4c7c3"},
                           {"range": [40, 70], "color": "#fde8b6"},
                           {"range": [70, 100], "color": "#c9e6c1"},
                       ]},
            ))
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            if pred < 40:
                interp = "This falls in the **lower** range of scores observed in this dataset."
            elif pred < 70:
                interp = "This falls in the **middle** range of scores observed in this dataset."
            else:
                interp = "This falls in the **higher** range of scores observed in this dataset."
            st.write(interp)
            st.caption(
                "Interpretation is relative to the score distribution in this dataset only "
                "(min={:.0f}, median={:.0f}, max={:.0f}). It is not a clinical threshold."
                .format(df_clean[TARGET].min(), df_clean[TARGET].median(), df_clean[TARGET].max())
            )
        st.info("Go to **Prediction Explanation** to see which inputs influenced this result.")

# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.title("📈 Model Performance")

    comp_df = pd.DataFrame(metrics["baseline_comparison"]).T
    comp_df = comp_df.sort_values("R2", ascending=False)
    st.subheader("Model comparison table")
    st.dataframe(comp_df, use_container_width=True)

    st.subheader(f"Best model: {metrics['best_baseline_model']} (tuned)")
    tm = metrics["tuned_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", tm["MAE"])
    c2.metric("RMSE", tm["RMSE"])
    c3.metric("R²", tm["R2"])
    c4.metric("Improvement vs baseline", f"{tm['R2'] - metrics['baseline_r2_of_best']:+.4f}")

    st.subheader("Actual vs Predicted (test set)")
    actual = metrics["test_actual"]
    predicted = metrics["test_predicted"]
    fig = px.scatter(x=actual, y=predicted, labels={"x": "Actual", "y": "Predicted"},
                      title="Actual vs Predicted Mental Health Score", opacity=0.6)
    fig.add_shape(type="line", x0=min(actual), y0=min(actual), x1=max(actual), y1=max(actual),
                  line=dict(color="red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Residual plot")
    residuals = np.array(actual) - np.array(predicted)
    fig = px.scatter(x=predicted, y=residuals, labels={"x": "Predicted", "y": "Residual"},
                      title="Residuals vs Predicted", opacity=0.6)
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature importance (global)")
    fi = pd.Series(metrics["feature_importance"]).sort_values(ascending=True).tail(12)
    fig = px.bar(x=fi.values, y=fi.index, orientation="h",
                 title="Top feature importances", labels={"x": "Importance", "y": "Feature"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Feature importance reflects how much the model relies on each input to make "
        "predictions. It does **not** imply that a feature medically causes a "
        "particular mental health outcome."
    )

# ============================================================
# PREDICTION EXPLANATION
# ============================================================
elif page == "Prediction Explanation":
    st.title("🔬 Prediction Explanation")

    if "last_pred" not in st.session_state:
        st.info("Make a prediction on the **Prediction** page first, then come back here.")
    else:
        st.metric("Last predicted score", f"{st.session_state['last_pred']:.1f}")
        st.subheader("Your inputs")
        display_df = st.session_state["last_input"].T.rename(columns={0: "Value"}); display_df["Value"] = display_df["Value"].astype(str); st.dataframe(display_df,
                     use_container_width=True)

        st.subheader("Which factors drive predictions in general")
        fi = pd.Series(metrics["feature_importance"]).sort_values(ascending=True).tail(10)
        fig = px.bar(x=fi.values, y=fi.index, orientation="h",
                     title="Global feature importance (model-wide, not specific to your inputs)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This is a global explanation of the trained model's behavior. Building a "
            "true per-prediction SHAP breakdown for a linear model reduces to each "
            "feature's (value - mean) x coefficient; the bar chart above shows which "
            "coefficients matter most overall."
        )

# ============================================================
# INSIGHTS
# ============================================================
elif page == "Insights":
    st.title("💡 Automated Insights")

    corr = df_clean[RAW_NUMERIC_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    strongest_pos = corr.idxmax()
    strongest_neg = corr.idxmin()

    c1, c2 = st.columns(2)
    c1.metric("Strongest positive association", strongest_pos, f"r = {corr.max():.2f}")
    c2.metric("Strongest negative association", strongest_neg, f"r = {corr.min():.2f}")

    st.subheader("All correlations with target")
    fig = px.bar(x=corr.values, y=corr.index, orientation="h",
                 title="Correlation of each feature with Mental Health Score",
                 color=corr.values, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score distribution by Employment Status")
    fig = px.violin(df_clean, x="Employment_Status", y=TARGET, box=True, points="outliers")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score distribution by Gender")
    fig = px.violin(df_clean, x="Gender", y=TARGET, box=True, points="outliers")
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "These are statistical associations in this dataset only. They are **not** "
        "causal or medical claims about what determines mental wellbeing."
    )
