# app.py
# ==============================================
# Streamlit Car Price Prediction App
# Author: Muhammad Umar Farooqi
# Date: 2025
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ============================================================
# SETUP
# ============================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
)

# Background function (fetch online image)
def add_bg_from_url(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img_data = response.content
            b64 = base64.b64encode(img_data).decode()
            css = f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{b64}");
                background-size: cover;
                background-attachment: fixed;
            }}
            </style>
            """
            st.markdown(css, unsafe_allow_html=True)
    except Exception as e:
        st.warning("Could not load background image.")

# Example free background
bg_url = "https://images.unsplash.com/photo-1502877338535-766e1452684a"
add_bg_from_url(bg_url)

# ============================================================
# HEADER
# ============================================================

st.title("Car Price Prediction App")
st.markdown(
    """
    This app predicts the **selling price of used cars**  
    using a Machine Learning model trained on past sales data.  
    Adjust the inputs below and hit **Predict**.
    """
)

# Load trained model
@st.cache_resource
def load_model(path="car_price_pipeline_model.joblib"):
    return joblib.load(path)

model = load_model()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("About")
st.sidebar.info(
    """
    ### Car Price Predictor  
    - Built with **Streamlit + Scikit-Learn**  
    - Author: *Muhammad Umar Farooqi*  
    - Dataset: Used Cars Dataset (301 rows)  
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Model: Linear Regression / RandomForest / ElasticNet")

# ============================================================
# INPUT FORM
# ============================================================

st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    car_name = st.text_input("Car Name", "swift vxi")
    year = st.number_input("Year of Manufacture", min_value=1995, max_value=2025, value=2016)
    present_price = st.number_input("Present Price (Lakhs)", min_value=0.5, max_value=50.0, value=6.0, step=0.5)
    driven_kms = st.number_input("Driven KMs", min_value=100, max_value=200000, value=42000, step=500)

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
    transmission = st.radio("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

# ============================================================
# FEATURE ENGINEERING (same as training pipeline)
# ============================================================

def extract_brand(name: str) -> str:
    if pd.isna(name): 
        return "other"
    brand = name.split()[0].lower()
    brand = "".join([c for c in brand if c.isalnum()])
    return brand if brand else "other"

def add_features(df_in: pd.DataFrame, snapshot_year: int = 2025) -> pd.DataFrame:
    df2 = df_in.copy()
    df2["Brand"] = df2["Car_Name"].astype(str).map(extract_brand)
    counts = df2["Brand"].value_counts()
    rare_brands = counts[counts < 5].index
    df2.loc[df2["Brand"].isin(rare_brands), "Brand"] = "other"
    df2["Age"] = (snapshot_year - df2["Year"]).clip(lower=0)
    df2["Driven_kms_log"] = np.log1p(df2["Driven_kms"])
    df2["price_per_age"] = df2["Present_Price"] / (df2["Age"] + 1)
    df2 = df2.drop(columns=["Car_Name", "Year"])
    return df2

def predict_price(example: dict):
    df_one = pd.DataFrame([example])
    df_one_fe = add_features(df_one)
    return float(model.predict(df_one_fe)[0])

# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("Predict Price"):
    user_input = {
        "Car_Name": car_name,
        "Year": year,
        "Selling_Price": 0, # placeholder
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Fuel_Type": fuel_type,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Owner": owner
    }
    with st.spinner("Predicting..."):
        time.sleep(1.5)
        prediction = predict_price(user_input)

    st.success(f"Estimated Selling Price: **{prediction:.2f} Lakhs**")

    # Plot for fun
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=["Present Price","Predicted Sell Price"], y=[present_price,prediction], ax=ax, palette="mako")
    ax.set_ylabel("Price (Lakhs)")
    st.pyplot(fig)

# ============================================================
# EXTRA FEATURES
# ============================================================

st.markdown("---")
st.subheader("Notes & Instructions")
st.markdown(
    """
    - Prices are predicted based on training data (~300 cars).  
    - Results may not be perfect, but good for demo purposes.  
    - Model considers **Brand, Age, KM Driven, Fuel, Transmission, Owner**.  
    """
)

st.markdown("---")
st.caption("Made with Love By Umar")
