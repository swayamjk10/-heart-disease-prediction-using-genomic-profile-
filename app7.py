# app.py

import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('heart_disease_model_9.pkl')
scaler = joblib.load('scaler9.pkl')

# Define a function to make predictions
def predict_heart_disease(age_years, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    # Convert age from years to days
    age_days = age_years * 365
    
    # Create BMI
    bmi = weight / (height / 100) ** 2
    
    # Prepare the feature vector for prediction (including BMI)
    features = np.array([[age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi]])
    
    # Ensure all features match the scaler's expected input
    features = scaler.transform(features)  # Scale features
    
    # Predict the risk
    risk = model.predict(features)
    return risk[0]

# Streamlit App
st.title("Heart Disease Risk Prediction")

# Collect user input
age_years = st.number_input("Age (in years)", min_value=1, max_value=100, step=1)
gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
height = st.number_input("Height (in cm)", min_value=100, max_value=250, step=1)
weight = st.number_input("Weight (in kg)", min_value=30, max_value=200, step=1)
ap_hi = st.number_input("Systolic blood pressure", min_value=90, max_value=200, step=1)
ap_lo = st.number_input("Diastolic blood pressure", min_value=60, max_value=130, step=1)
cholesterol = st.selectbox("Cholesterol level", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
gluc = st.selectbox("Glucose level", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
smoke = st.selectbox("Do you smoke?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
alco = st.selectbox("Do you consume alcohol?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
active = st.selectbox("Are you physically active?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Prediction button
if st.button("Predict"):
    result = predict_heart_disease(age_years, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
    st.write("Prediction:", result)
