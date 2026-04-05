import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.markdown("### Predict risk based on clinical parameters")

# -------------------------
# 📘 Feature Descriptions
# -------------------------
with st.expander("ℹ️ Feature Descriptions"):
    st.markdown("""
    - **Age**: Age of the patient (years)  
    - **Sex**: Male (M) or Female (F)  
    - **Chest Pain Type**:
        - ATA: Atypical Angina  
        - NAP: Non-Anginal Pain  
        - ASY: Asymptomatic  
        - TA: Typical Angina  
    - **RestingBP**: Resting blood pressure (mm Hg)  
    - **Cholesterol**: Serum cholesterol (mg/dl)  
    - **FastingBS**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
    - **RestingECG**:
        - Normal  
        - ST: ST-T wave abnormality  
        - LVH: Left ventricular hypertrophy  
    - **MaxHR**: Maximum heart rate achieved  
    - **ExerciseAngina**: Exercise-induced angina (Y/N)  
    - **Oldpeak**: ST depression induced by exercise  
    - **ST_Slope**:
        - Up: Upsloping (better condition)  
        - Flat: Flat (moderate risk)  
        - Down: Downsloping (high risk)  
    """)

# -------------------------
# 🧾 Input Section
# -------------------------
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 80, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    restingBP = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 400, 200)
    fastingBS = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    restingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    maxHR = st.number_input("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -------------------------
# 🔄 Preprocessing
# -------------------------
input_data = {
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "RestingBP": restingBP,
    "Cholesterol": cholesterol,
    "FastingBS": fastingBS,
    "RestingECG": restingECG,
    "MaxHR": maxHR,
    "ExerciseAngina": exercise_angina,
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope
}

input_df = pd.DataFrame([input_data])
input_df = pd.get_dummies(input_df)

for col in columns:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[columns]
input_scaled = scaler.transform(input_df)

# -------------------------
# 🔮 Prediction
# -------------------------
if st.button("Predict"):
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prob > 0.4:
        st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nProbability: {prob:.2f}")

# -------------------------
# 📊 Feature Importance
# -------------------------
st.subheader("📊 Feature Importance")

importances = model.feature_importances_
feat_series = pd.Series(importances, index=columns)

top_features = feat_series.sort_values(ascending=False).head(10)

fig, ax = plt.subplots()
top_features.sort_values().plot(kind='barh', ax=ax)
ax.set_title("Top 10 Important Features")

st.pyplot(fig)