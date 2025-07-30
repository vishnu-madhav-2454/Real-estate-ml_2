import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title='Gurgaon Property Price Predictor', layout="centered")

# Load all required files
with open('pickle files/df.pkl', 'rb') as f:
    df = pickle.load(f)

with open('pickle files/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('pickle files/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pickle files/ordinal_encoder.pkl', 'rb') as f:
    ordinal_encoder = pickle.load(f)

with open('pickle files/onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

# Fixed MAE in log1p scale
mae = 0.22

# Title
st.title('🏠 Gurgaon Property Price Predictor')

st.markdown("### Enter the property details below:")

# Inputs
property_type = st.selectbox('Property Type', sorted(df['property_type'].unique()))
sector = st.selectbox('Sector', sorted(df['sector'].unique()))
bedrooms = float(st.selectbox("Number of Bedrooms", sorted(df['bedRoom'].unique())))
bathrooms = float(st.selectbox("Number of Bathrooms", sorted(df['bathroom'].unique())))
balcony = st.selectbox("Number of Balconies", sorted(df['balcony'].unique()))
age = st.selectbox("Property Age", sorted(df['agePossession'].unique()))
built_up_area = float(st.number_input("Built-up Area (sq ft)", min_value=100.0))
servant_room = float(st.selectbox("Servant Room Present?", [0.0, 1.0]))
store_room = float(st.selectbox("Store Room Present?", [0.0, 1.0]))
furnishing_type = st.selectbox("Furnishing Type", sorted(df['furnishing_type'].unique()))
luxury_category = st.selectbox("Luxury Category", sorted(df['luxury_category'].unique()))
floor_category = st.selectbox("Floor Category", sorted(df['floor_category'].unique()))

if st.button('Predict Price'):
    input_dict = {
        'property_type': property_type,
        'sector': sector,
        'bedRoom': bedrooms,
        'bathroom': bathrooms,
        'balcony': balcony,
        'agePossession': age,
        'built_up_area': built_up_area,
        'servant room': servant_room,
        'store room': store_room,
        'furnishing_type': furnishing_type,
        'luxury_category': luxury_category,
        'floor_category': floor_category
    }

    input_df = pd.DataFrame([input_dict])

    # Column groups
    num_cols = ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']
    ordinal_cols = ['property_type', 'balcony', 'furnishing_type', 'luxury_category', 'floor_category']
    onehot_cols = ['sector', 'agePossession']

    try:
        # Preprocess input
        X_num = scaler.transform(input_df[num_cols])
        X_ord = ordinal_encoder.transform(input_df[ordinal_cols])
        X_oh = onehot_encoder.transform(input_df[onehot_cols])
        X_final = np.hstack([X_num, X_ord, X_oh])

        # Predict log1p price
        pred_log_price = model.predict(X_final)[0]
        pred_price = np.expm1(pred_log_price)

        # Calculate range using log1p MAE
        lower_price = np.expm1(pred_log_price) + mae
        upper_price = np.expm1(pred_log_price) - mae

        st.success(f"🏷️ Estimated Price: ₹ {pred_price:,.3f} CR")
        st.info(f"🔍 Price Range: ₹ {lower_price:,.3f} CR — ₹ {upper_price:,.3f} CR")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
