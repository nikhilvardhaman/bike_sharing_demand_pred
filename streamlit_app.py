import streamlit as st
import pandas as pd
from datetime import datetime
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Title and Description
st.title("Bike Sharing Demand Prediction")
st.write("Enter the details below to predict the bike-sharing demand.")

# Input Form
with st.form("prediction_form"):
    # Date and Time Input
    selected_date = st.date_input("Select Date", value=datetime(2011, 1, 1).date(), min_value=datetime(2011, 1, 1).date(), max_value=datetime(2012, 12, 31).date())
    selected_time = st.time_input("Select Time", value=datetime(2011, 1, 1, 0, 0).time())

    # Combine Date and Time
    datetime_input = datetime.combine(selected_date, selected_time).strftime('%Y-%m-%d %H:%M:%S')

    season = st.selectbox("Season", options=[1, 2, 3, 4], format_func=lambda x: ["Spring", "Summer", "Fall", "Winter"][x - 1])
    holiday = st.selectbox("Is it a Holiday?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    workingday = st.selectbox("Is it a Working Day?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    weather = st.selectbox(
        "Weather",
        options=[1, 2, 3, 4],
        format_func=lambda x: [
            "Clear or Partly Cloudy",
            "Mist",
            "Light Snow or Rain",
            "Heavy Rain or Snow"
        ][x - 1],
    )
    temp = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
    atemp = st.number_input("Feels-like Temperature (°C)", min_value=0.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100)
    windspeed = st.number_input("Windspeed (m/s)", min_value=0.0, step=0.1)

    # Submit Button
    submitted = st.form_submit_button("Predict Demand")

# Prediction Logic
if submitted:
    try:
        # Create a data object
        data = CustomData(
            datetime=datetime_input,
            season=season,
            holiday=holiday,
            workingday=workingday,
            weather=weather,
            temp=temp,
            atemp=atemp,
            humidity=humidity,
            windspeed=windspeed
        )
        pred_df = data.get_data_as_data_frame()

        # Predict using the pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Display Results
        st.success(f"The predicted bike-sharing demand is: {results[0]}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
