import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS history (
    username TEXT,
    age INTEGER,
    probability REAL,
    timestamp TEXT
)
""")

conn.commit()

# ---------------- LOGIN SYSTEM ----------------
st.sidebar.title("User Authentication")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if choice == "Register":
    if st.sidebar.button("Register"):
        try:
            c.execute("INSERT INTO users VALUES (?, ?)", (username, password))
            conn.commit()
            st.sidebar.success("User created successfully!")
        except:
            st.sidebar.error("Username already exists")

if choice == "Login":
    if st.sidebar.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        data = c.fetchone()

        if data:
            st.session_state["user"] = username
            st.sidebar.success("Logged in successfully!")
        else:
            st.sidebar.error("Invalid credentials")

# ---------------- PROTECT APP ----------------
if "user" not in st.session_state:
    st.warning("Please login to access the app")
    st.stop()

current_user = st.session_state["user"]

# ---------------- UI ----------------
st.title("❤️ Heart Health Risk Checker")
st.markdown("### Predict your heart disease risk easily")

st.info(f"Logged in as: {current_user}")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 20, 80, 40)

    sex = st.selectbox("Gender", ["Male", "Female"])

    chest_pain = st.selectbox(
        "Type of chest pain",
        [
            "Mild discomfort (ATA)",
            "Non-heart pain (NAP)",
            "No symptoms but risk (ASY)",
            "Severe chest pain (TA)"
        ]
    )

    restingBP = st.number_input("Resting blood pressure", 80, 200, 120)

    cholesterol = st.number_input("Cholesterol level", 100, 400, 200)

    fastingBS = st.selectbox("High blood sugar?", ["No", "Yes"])

with col2:
    restingECG = st.selectbox(
        "ECG result",
        ["Normal", "ST abnormality", "LVH"]
    )

    maxHR = st.number_input("Maximum heart rate", 60, 220, 150)

    exercise_angina = st.selectbox(
        "Chest pain during exercise?",
        ["No", "Yes"]
    )

    oldpeak = st.number_input(
        "Exercise discomfort level (0 = none)",
        0.0, 6.0, 1.0
    )

    st_slope = st.selectbox(
        "Heart response during exercise",
        ["Normal (Up)", "Flat (Risk)", "Down (High Risk)"]
    )

# ---------------- CONVERT VALUES ----------------
sex = "M" if sex == "Male" else "F"

chest_map = {
    "Mild discomfort (ATA)": "ATA",
    "Non-heart pain (NAP)": "NAP",
    "No symptoms but risk (ASY)": "ASY",
    "Severe chest pain (TA)": "TA"
}
chest_pain = chest_map[chest_pain]

fastingBS = 1 if fastingBS == "Yes" else 0

ecg_map = {
    "Normal": "Normal",
    "ST abnormality": "ST",
    "LVH": "LVH"
}
restingECG = ecg_map[restingECG]

exercise_angina = "Y" if exercise_angina == "Yes" else "N"

slope_map = {
    "Normal (Up)": "Up",
    "Flat (Risk)": "Flat",
    "Down (High Risk)": "Down"
}
st_slope = slope_map[st_slope]

# ---------------- PREPROCESS ----------------
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

# ---------------- PREDICTION ----------------
if st.button("Check Heart Risk"):
    prob = model.predict_proba(input_scaled)[0][1]

    # Save to database
    c.execute(
        "INSERT INTO history VALUES (?, ?, ?, ?)",
        (current_user, age, float(prob), datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    conn.commit()

    st.subheader("Result")

    if prob < 0.3:
        st.success(f"🟢 Low Risk (Probability: {prob:.2f})")

    elif prob < 0.6:
        st.warning(f"🟡 Moderate Risk (Probability: {prob:.2f})")

    else:
        st.error(f"🔴 High Risk (Probability: {prob:.2f})")

    st.progress(int(prob * 100))

    # ---------------- SUGGESTIONS ----------------
    st.subheader("💡 What Should You Do?")

    if prob < 0.3:
        st.info("Maintain healthy lifestyle and regular exercise.")

    elif prob < 0.6:
        st.warning("Improve diet, monitor BP, increase activity.")

    else:
        st.error("Consult a doctor immediately.")

# ---------------- HISTORY ----------------
st.subheader("📊 Your Risk History")

c.execute("SELECT age, probability, timestamp FROM history WHERE username=?", (current_user,))
data = c.fetchall()

if data:
    df = pd.DataFrame(data, columns=["Age", "Risk", "Time"])
    st.dataframe(df)

    st.line_chart(df["Risk"])
else:
    st.write("No history yet")

# ---------------- DISCLAIMER ----------------
st.caption("⚠️ This tool is for educational purposes only and not medical advice.")