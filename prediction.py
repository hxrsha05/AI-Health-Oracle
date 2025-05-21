import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random

# Load model and encoders
model = joblib.load("E:/NaaN Mudhalvan/random_forest_model.pkl")
label_encoders = joblib.load("E:/NaaN Mudhalvan/label_encoders.pkl")
scaler = joblib.load("E:/NaaN Mudhalvan/scaler.pkl")
feature_order = joblib.load("E:/NaaN Mudhalvan/feature_order.pkl")

data = pd.read_csv("E:/NaaN Mudhalvan/Disease_symptom_and_patient_profile_dataset.csv")
data.columns = data.columns.str.strip()

st.set_page_config(page_title="🧠 AI Health Oracle", layout="centered")
st.title("🧠 AI Health Oracle")
st.markdown("### Diagnose Your Condition Based on Symptoms & Profile")
st.image("https://cdn.pixabay.com/photo/2017/03/15/12/42/brain-2146817_1280.png", width=150)

st.markdown("#### 👇 Fill your health details:")
user_input = {}

for col in feature_order:
    if data[col].dtype == object or data[col].nunique() <= 10:
        options = sorted(data[col].dropna().unique())
        user_input[col] = st.selectbox(f"{col}:", options)
    else:
        min_val = int(data[col].min())
        max_val = int(data[col].max())
        user_input[col] = st.slider(f"{col}:", min_val, max_val, step=1)

# Encode
input_df = pd.DataFrame([user_input])
for col in label_encoders:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Align column order
input_df = input_df[feature_order]

# Scale
input_scaled = scaler.transform(input_df)

if st.button("🩺 Diagnose"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🧬 Predicted Disease: **{prediction}**")
    st.balloons()

    st.info("💡 Health Tip: " + random.choice([
        "💧 Drink more water!",
        "🥗 Eat more fiber!",
        "🛌 Get 7-8 hours of sleep!",
        "🏃 Exercise daily!",
        "🧘 Practice mindfulness!"
    ]))
