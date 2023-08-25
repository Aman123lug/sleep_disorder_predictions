import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle

stn = StandardScaler()
label_encoder = LabelEncoder()

st.header('Sleep-Disorder-Prediction 🚀')

gender = st.selectbox('Enter your gender',("Male", "Female"))
age = st.number_input("enter your age")
occupation = st.selectbox("Enter your Occupation", ("Doctor"))
sleep_duration = st.number_input("enter your Sleep Duration in Hrs")
quality_sleep = st.number_input("enter your Quality of Sleep in Hrs")
physical_level = st.number_input("enter your Physical Activity Level")
stress_level = st.number_input("enter your Stress level")
bmi = st.number_input("enter your BMI category")
blood_presure = st.number_input("enter your Blood presure")
heart_rate = st.number_input("enter your heart rate")
daily_steps = st.number_input("enter your daily steps")
systolic_bp = st.number_input("enter your Systolic BP")
diastolic_bp = st.number_input("enter your Diastolic_BP")

# scaling part
daily_steps = stn.fit_transform(daily_steps)

# encoder 
gender = label_encoder.fit_transform(gender)
occupation = label_encoder.fit_transform(occupation)
bmi = label_encoder.fit_transform(bmi)


final_input = [gender, age, occupation, sleep_duration, quality_sleep, physical_level, stress_level, bmi, blood_presure, heart_rate, systolic_bp, diastolic_bp]

button = st.button("Predict")
if button:
    model = pickle.load("model/model.pkl")
    sleep_disorder = model.predict(final_input)
    if sleep_disorder == 0:
        st.write("You are healty")
        
    elif sleep_disorder == 1:
        st.write("Sleep Apnea")

    else:
        st.write("Insomnia")    
    
