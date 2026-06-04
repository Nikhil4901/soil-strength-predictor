import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Set page configuration
st.set_page_config(page_title="Soil Engineering Predictor", layout="wide")

# Get the absolute path of the directory where GUI.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models and scalers using absolute paths to avoid directory errors
@st.cache_resource
def load_assets():
    try:
        model_cbr_path = os.path.join(BASE_DIR, 'model_cbr.pkl')
        model_ucs_path = os.path.join(BASE_DIR, 'model_ucs.pkl')
        scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
        
        model_cbr = pickle.load(open(model_cbr_path, 'rb'))
        model_ucs = pickle.load(open(model_ucs_path, 'rb'))
        scaler = pickle.load(open(scaler_path, 'rb'))
        
        # FIX: Check if models are tuples and extract the actual model
        if isinstance(model_cbr, tuple):
            model_cbr = model_cbr[0]  # Extract first element
            st.warning("⚠️ model_cbr.pkl contained a tuple. Extracted model from tuple.")
        
        if isinstance(model_ucs, tuple):
            model_ucs = model_ucs[0]  # Extract first element
            st.warning("⚠️ model_ucs.pkl contained a tuple. Extracted model from tuple.")
        
        return model_cbr, model_ucs, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. {e}")
        st.info(f"Make sure your pickle files are placed in this folder: {BASE_DIR}")
        return None, None, None

model_cbr, model_ucs, scaler = load_assets()
st.title("🏗️ Geotechnical Engineering Strength Predictor")
st.markdown("Choose your target parameter below to calculate soil stabilization design criteria.")

if model_cbr and model_ucs and scaler:
    # 1. Selection Interface
    prediction_target = st.radio(
        "**Select the Target Metric to Predict:**",
        options=["California Bearing Ratio (CBR)", "Unconfined Compressive Strength (UCS)"],
        horizontal=True
    )
    
    st.divider()

    with st.form("dynamic_input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧪 Additive Mixture Content")
            fly_ash = st.number_input("Fly Ash (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            lime = st.number_input("Lime (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            cement = st.number_input("Cement (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            
        with col2:
            st.subheader("📊 Base Physical Properties")
            ll = st.number_input("Liquid Limit LL (%)", min_value=0.0, max_value=100.0, value=35.0, step=0.1)
            pl = st.number_input("Plastic Limit PL (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
            omc = st.number_input("Optimum Moisture Content OMC (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)
            mdd = st.number_input("Maximum Dry Density MDD (g/cm³)", min_value=0.0, max_value=3.0, value=1.8, step=0.01)
            
            # Context-Aware Input Fields: If predicting UCS, it expects a CBR input feature as well!
            if prediction_target == "Unconfined Compressive Strength (UCS)":
                st.markdown("---")
                st.markdown("**Additional feature required for UCS model:**")
                cbr_input = st.number_input("Known/Estimated CBR (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)

        submit = st.form_submit_button(f"Calculate {prediction_target}")

    if submit:
        try:
            if prediction_target == "California Bearing Ratio (CBR)":
                # --- CASE A: CBR MODE ---
                # Build the 9-element array using dummies for column index 7 and 8
                # Layout: [Fly Ash, Lime, Cement, LL, PL, OMC, MDD, CBR_dummy, UCS_dummy]
                dummy_row = np.array([[fly_ash, lime, cement, ll, pl, omc, mdd, 0.0, 0.0]])
                
                # Transform using the shared scaler
                scaled_row = scaler.transform(dummy_row)
                
                # Slice first 7 features for the CBR model
                input_scaled_cbr = scaled_row[:, :7]
                pred_cbr_scaled = model_cbr.predict(input_scaled_cbr)[0]
                
                # Assign back to matrix index [7] and perform inverse transform to read true metric value
                scaled_row[0, 7] = pred_cbr_scaled
                unscaled_matrix = scaler.inverse_transform(scaled_row)
                final_cbr = max(0.0, unscaled_matrix[0, 7])
                
                # Display Results
                st.success("CBR Calculation Completed!")
                st.metric(label="🎯 Predicted California Bearing Ratio (CBR)", value=f"{final_cbr:.2f} %")
                
            else:
                # --- CASE B: UCS MODE ---
                # Build the 9-element array using user's explicit CBR input and a dummy for column index 8
                # Layout: [Fly Ash, Lime, Cement, LL, PL, OMC, MDD, User_CBR, UCS_dummy]
                dummy_row = np.array([[fly_ash, lime, cement, ll, pl, omc, mdd, cbr_input, 0.0]])
                
                # Transform using the shared scaler
                scaled_row = scaler.transform(dummy_row)
                
                # Slice first 8 features (which correctly includes the scaled version of the user-provided CBR)
                input_scaled_ucs = scaled_row[:, :8]
                pred_ucs_scaled = model_ucs.predict(input_scaled_ucs)[0]
                
                # Assign back to matrix index [8] and perform inverse transform to read true metric value
                scaled_row[0, 8] = pred_ucs_scaled
                unscaled_matrix = scaler.inverse_transform(scaled_row)
                final_ucs = max(0.0, unscaled_matrix[0, 8])
                
                # Display Results
                st.success("UCS Calculation Completed!")
                st.metric(label="💪 Predicted Unconfined Compressive Strength (UCS)", value=f"{final_ucs:.2f} kg/cm²")
                
        except Exception as e:
            st.error(f"An error occurred during computational scaling or prediction: {e}")
else:
    st.warning("Please ensure 'model_cbr.pkl', 'model_ucs.pkl', and 'scaler.pkl' are placed together in the target script folder.")