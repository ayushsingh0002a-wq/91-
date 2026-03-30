import streamlit as st
import pandas as pd
from utils import preprocess_data
from model import train_model, predict_next

st.title("🎯 Big Small Color Prediction Tool")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("📊 Raw Data", df.head())

    df = preprocess_data(df)

    size_model, color_model = train_model(df)

    last_number = st.number_input("Enter Last Number (0-9)", min_value=0, max_value=9)

    if st.button("Predict"):
        size, color = predict_next(size_model, color_model, last_number)

        st.success(f"Prediction: {size} | Color: {color}")
