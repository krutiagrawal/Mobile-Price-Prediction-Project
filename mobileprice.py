import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib


model = load_model("mobile_price_model.h5")
scaler = joblib.load("scaler.pkl")


st.title("Mobile Price Prediction")


battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=6000, step=100)
blue = st.selectbox("Bluetooth", ["No", "Yes"])
clock_speed = st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.0, step=0.1)
dual_sim = st.selectbox("Dual SIM", ["No", "Yes"])
fc = st.number_input("Front Camera (MP)", min_value=0, max_value=20, step=1)
four_g = st.selectbox("4G", ["No", "Yes"])
int_memory = st.number_input("Internal Memory (GB)", min_value=0, max_value=256, step=4)
m_deep = st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.0, step=0.1)
mobile_wt = st.number_input("Mobile Weight (gm)", min_value=50, max_value=300, step=10)
n_cores = st.number_input("Processor Cores", min_value=1, max_value=10, step=1)
pc = st.number_input("Primary Camera (MP)", min_value=0, max_value=48, step=1)
px_height = st.number_input("Pixel Resolution Height", min_value=0, max_value=3000, step=100)
px_width = st.number_input("Pixel Resolution Width", min_value=0, max_value=3000, step=100)
ram = st.number_input("RAM (MB)", min_value=512, max_value=8000, step=128)
sc_h = st.number_input("Screen Height (cm)", min_value=5, max_value=20, step=1)
sc_w = st.number_input("Screen Width (cm)", min_value=5, max_value=20, step=1)
talk_time = st.number_input("Talk Time (hours)", min_value=2, max_value=30, step=1)
three_g = st.selectbox("3G", ["No", "Yes"])
touch_screen = st.selectbox("Touch Screen", ["No", "Yes"])
wifi = st.selectbox("WiFi", ["No", "Yes"])

if st.button("Calculate Price Range"):
    blue = 1 if blue == "Yes" else 0
    dual_sim = 1 if dual_sim == "Yes" else 0
    four_g = 1 if four_g == "Yes" else 0
    three_g = 1 if three_g == "Yes" else 0
    touch_screen = 1 if touch_screen == "Yes" else 0
    wifi = 1 if wifi == "Yes" else 0

    input_data = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_deep,
                            mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time,
                            three_g, touch_screen, wifi]])

    input_data_scaled = scaler.transform(input_data)


    prediction = model.predict(input_data_scaled)
    predicted_price_range = np.argmax(prediction, axis=1)[0]


    price_ranges = ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]
    st.write("Predicted Price Range:", price_ranges[predicted_price_range])
