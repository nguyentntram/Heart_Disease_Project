import os
import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")
st.title("Heart Disease Risk Predictor")
st.write("Enter patient features to get a predicted probability of heart disease.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.pkl")
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please train the model first (run `python -m src.train_pipeline`).")
    st.stop()

pipe = load(MODEL_PATH)

# ===== Inputs (Cleveland coding) =====
st.subheader("Input Features")

age = st.number_input("age", min_value=1, max_value=120, value=54)
sex = st.selectbox("sex (1=male, 0=female)", options=[0, 1], index=1)

# cp: 1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic
cp = st.selectbox(
    "cp (chest pain type: 1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic)",
    options=[1, 2, 3, 4],
    index=2
)

trestbps = st.number_input("trestbps (resting blood pressure)", min_value=50, max_value=260, value=130)
chol = st.number_input("chol (serum cholesterol)", min_value=50, max_value=700, value=246)
fbs = st.selectbox("fbs (>120 mg/dl) (1/0)", options=[0, 1], index=0)
restecg = st.selectbox("restecg (0,1,2)", options=[0, 1, 2], index=0)
thalach = st.number_input("thalach (max heart rate achieved)", min_value=50, max_value=260, value=150)
exang = st.selectbox("exang (exercise induced angina) (1/0)", options=[0, 1], index=0)
oldpeak = st.number_input("oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.2, step=0.1, format="%.1f")

# slope: 1=upsloping, 2=flat, 3=downsloping
slope = st.selectbox("slope (1=up, 2=flat, 3=down)", options=[1, 2, 3], index=1)

ca = st.selectbox("ca (number of major vessels 0–3)", options=[0, 1, 2, 3], index=0)

# thal: {3=normal, 6=fixed defect, 7=reversible defect}; allow Unknown -> imputed by pipeline
thal_map = {
    "Unknown": None,
    "3 = normal": 3,
    "6 = fixed defect": 6,
    "7 = reversible defect": 7,
}
thal_label = st.selectbox("thal", options=list(thal_map.keys()), index=0)
thal = thal_map[thal_label]

# Decision threshold slider
threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)

if st.button("Predict"):
    row = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    X = pd.DataFrame([row])
    proba = pipe.predict_proba(X)[:, 1][0]
    pred = int(proba >= threshold)

    st.success(f"Predicted probability of heart disease: {proba:.3f}")
    st.write(f"Predicted class: **{pred}** (threshold {threshold:.2f})")
