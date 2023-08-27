import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle
import numpy as np

stn = StandardScaler()
label_encoder = LabelEncoder()

st.header('Sleep-Disorder-Prediction 🚀')

gender = st.selectbox('Enter your gender',("Male", "Female"))
age = int(st.number_input("enter your age"))
occupation = st.selectbox("Enter your Occupation", ["Docter", "Employee", "Software-engineer", "Sales-reprentater", "Teacher", "Nurse", "Engineer", "accountant", "Scientist", "Lawer", "Sale-Person", "Manager"])
sleep_duration = st.number_input("enter your Sleep Duration in Hrs")
quality_sleep = st.number_input("enter your Quality of Sleep in Hrs")
physical_level = st.number_input("enter your Physical Activity Level")
stress_level = st.number_input("enter your Stress level")
bmi = st.selectbox("enter your BMI category", ['Overweight', 'Normal', 'Obese', 'Under Weight'])
blood_presure = st.number_input("enter your weight in kgs")
heart_rate = st.number_input("enter your heart rate")
daily_steps = int(st.number_input("enter your daily steps"))
systolic_bp = st.number_input("enter your Systolic BP")
diastolic_bp = st.number_input("enter your Diastolic_BP")

print(type(daily_steps))
# scaling part
daily_steps = stn.fit_transform([[daily_steps]])

# encoder 
gender = label_encoder.fit_transform([gender])
occupation = label_encoder.fit_transform([occupation])
bmi = label_encoder.fit_transform([bmi])

final_input = np.array([gender, age, occupation, sleep_duration, quality_sleep, physical_level, stress_level, bmi, blood_presure, heart_rate, systolic_bp, diastolic_bp], dtype=object).reshape(1,12)

button = st.button("Predict") 
if button:
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    # model = pickle.loads(open("model.pkl", "rb"))
    sleep_disorder = model.predict(final_input)
    if sleep_disorder == 0:
        st.write("Insomnia")
        
    elif sleep_disorder == 1:
        st.write("Healthy")

    else:
        st.write("Sleep Apnea")