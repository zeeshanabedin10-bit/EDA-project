"""Streamlit web app for predicting house prices with the trained Ames model."""

import pandas as pd
import streamlit as st
from predict import DATA_PATH, predict_price

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 Ames House Price Predictor")
st.caption("Powered by Tuned Lasso Regression trained on the Ames Housing Dataset (R² = 0.9127)")

df_eng = pd.read_csv(DATA_PATH)
neighborhoods = ["(Dataset Median)"] + sorted(
    [c.replace("Neighborhood_", "") for c in df_eng.columns if c.startswith("Neighborhood_")]
)

col1, col2 = st.columns(2)
with col1:
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 7)
    gr_liv_area = st.number_input("Above-Grade Living Area (sq ft)", 400, 5000, 1800, step=50)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 1000, step=50)
with col2:
    garage_area = st.number_input("Garage Area (sq ft)", 0, 1500, 500, step=25)
    year_built = st.slider("Year Built", 1872, 2010, 2000)
    full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3], index=2)

neighborhood = st.selectbox("Neighborhood", neighborhoods)
selected_nb = None if neighborhood == "(Dataset Median)" else neighborhood

if st.button("Predict Sale Price", type="primary", use_container_width=True):
    price = predict_price(
        overall_qual=overall_qual,
        gr_liv_area=gr_liv_area,
        total_bsmt_sf=total_bsmt_sf,
        garage_area=garage_area,
        year_built=year_built,
        full_bath=full_bath,
        neighborhood=selected_nb,
    )
    st.success(f"### Estimated Sale Price: **${price:,.0f}**")
    st.info("Note: Unspecified features default to dataset medians from `train_engineered.csv`.")
