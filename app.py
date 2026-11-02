import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load trained artifacts
model = joblib.load('best_bean_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Dry Bean Classification App", page_icon="🌱", layout="wide")

st.title("🌱 Dry Bean Type Classification System")
st.markdown("Provide the physical measurements of the bean below to predict its exact agricultural class.")

st.sidebar.header("Input Bean Measurements")

def user_input_features():
    area = st.sidebar.number_input("Area (A)", min_value=1000.0, max_value=200000.0, value=28395.0)
    perimeter = st.sidebar.number_input("Perimeter (P)", min_value=100.0, max_value=2000.0, value=610.29)
    major_axis = st.sidebar.number_input("Major Axis Length (L)", min_value=50.0, max_value=500.0, value=208.18)
    minor_axis = st.sidebar.number_input("Minor Axis Length (l)", min_value=30.0, max_value=400.0, value=173.89)
    aspect_ratio = st.sidebar.number_input("Aspect Ratio (K)", min_value=1.0, max_value=5.0, value=1.19)
    eccentricity = st.sidebar.number_input("Eccentricity (Ec)", min_value=0.0, max_value=1.0, value=0.55)
    convex_area = st.sidebar.number_input("Convex Area (C)", min_value=1000.0, max_value=200000.0, value=28715.0)
    equiv_diameter = st.sidebar.number_input("Equivalent Diameter (Ed)", min_value=50.0, max_value=500.0, value=190.14)
    extent = st.sidebar.number_input("Extent (Ex)", min_value=0.0, max_value=1.0, value=0.76)
    solidity = st.sidebar.number_input("Solidity (S)", min_value=0.0, max_value=1.0, value=0.99)
    roundness = st.sidebar.number_input("Roundness (R)", min_value=0.0, max_value=1.0, value=0.96)
    compactness = st.sidebar.number_input("Compactness (CO)", min_value=0.0, max_value=1.0, value=0.91)
    sf1 = st.sidebar.number_input("ShapeFactor1 (SF1)", min_value=0.0, max_value=0.1, value=0.0073)
    sf2 = st.sidebar.number_input("ShapeFactor2 (SF2)", min_value=0.0, max_value=0.1, value=0.0031)
    sf3 = st.sidebar.number_input("ShapeFactor3 (SF3)", min_value=0.0, max_value=1.0, value=0.83)
    sf4 = st.sidebar.number_input("ShapeFactor4 (SF4)", min_value=0.0, max_value=1.0, value=0.99)
    
    data = {
        'Area': area, 'Perimeter': perimeter, 'MajorAxisLength': major_axis,
        'MinorAxisLength': minor_axis, 'AspectRation': aspect_ratio, 'Eccentricity': eccentricity,
        'ConvexArea': convex_area, 'EquivDiameter': equiv_diameter, 'Extent': extent,
        'Solidity': solidity, 'roundness': roundness, 'Compactness': compactness,
        'ShapeFactor1': sf1, 'ShapeFactor2': sf2, 'ShapeFactor3': sf3, 'ShapeFactor4': sf4
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input_df)

if st.button("Predict Bean Type"):
    # Preprocess and predict
    scaled_input = scaler.transform(input_df)
    prediction_encoded = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    
    predicted_class = le.inverse_transform(prediction_encoded)[0]
    confidence = np.max(prediction_proba) * 100
    
    st.success(f"### Predicted Bean Type: **{predicted_class}** (Confidence: {confidence:.2f}%)")