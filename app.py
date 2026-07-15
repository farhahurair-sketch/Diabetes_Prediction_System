import streamlit as st
import pandas as pd
import joblib

# Load Model and Scaler
model = joblib.load("best_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page Title
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺")

st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient details below to predict whether the patient is diabetic.")

st.header("Patient Information")

# Numerical Inputs
age = st.number_input("Age", min_value=1, max_value=100, value=30)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=10.0, value=5.5)

blood_glucose = st.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=400,
    value=120
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

smoking = st.selectbox(
    "Smoking History",
    [
        "No Info",
        "current",
        "ever",
        "former",
        "never",
        "not current"
    ]
)
# Prediction Button
if st.button("Predict"):

    # One-Hot Encoding
    gender_male = 1 if gender == "Male" else 0
    gender_other = 1 if gender == "Other" else 0

    smoking_current = 1 if smoking == "current" else 0
    smoking_ever = 1 if smoking == "ever" else 0
    smoking_former = 1 if smoking == "former" else 0
    smoking_never = 1 if smoking == "never" else 0
    smoking_not_current = 1 if smoking == "not current" else 0

    # Create Input DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [blood_glucose],
        "gender_Male": [gender_male],
        "gender_Other": [gender_other],
        "smoking_history_current": [smoking_current],
        "smoking_history_ever": [smoking_ever],
        "smoking_history_former": [smoking_former],
        "smoking_history_never": [smoking_never],
        "smoking_history_not current": [smoking_not_current]
    })

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Prediction Probability
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ The patient is likely to have Diabetes.")
    else:
        st.success("✅ The patient is not likely to have Diabetes.")

    st.write(f"**Probability of No Diabetes:** {probability[0]*100:.2f}%")
    st.write(f"**Probability of Diabetes:** {probability[1]*100:.2f}%")