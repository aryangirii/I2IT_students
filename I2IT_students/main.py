import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🏖 Travel Income Category Prediction")
st.write("Enter the details below to predict the **Income Category**.")

# --- User Inputs ---
# Adjust feature names to match your dataset columns (excluding MonthlyIncome & IncomeCategory)
age = st.number_input("Age", min_value=18, max_value=100, step=1)
education = st.selectbox("Education Level", ["High School", "Graduate", "Postgraduate"])
gender = st.selectbox("Gender", ["Male", "Female"])
experience = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
travel_freq = st.selectbox("Travel Frequency", ["Rarely", "Sometimes", "Often"])

# Convert inputs to DataFrame (structure must match training data before scaling)
input_df = pd.DataFrame({
    "Age": [age],
    "Education": [education],
    "Gender": [gender],
    "Experience": [experience],
    "Travel_Frequency": [travel_freq]
})

# One-hot encode like training
input_df = pd.get_dummies(input_df)

# Ensure all columns match the training data structure
# Create an empty DataFrame with all training columns
expected_cols = pd.get_dummies(pd.read_csv("Travel.csv").drop(['MonthlyIncome', 'IncomeCategory'], axis=1)).columns
input_df = input_df.reindex(columns=expected_cols, fill_value=0)

# Scale the data
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict Income Category"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"Predicted Income Category: **{prediction}**")
