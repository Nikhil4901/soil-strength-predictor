import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Soil Engineering Predictor", layout="wide")

# Initialize session state to store history for CSV export
if 'history' not in st.session_state:
    st.session_state.history = []

# Load models and scalers
@st.cache_resource
def load_assets():
    try:
        model_cbr = pickle.load(open('model_cbr.pkl', 'rb'))
        model_ucs = pickle.load(open('model_ucs.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model_cbr, model_ucs, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. {e}")
        return None, None, None

model_cbr, model_ucs, scaler = load_assets()

st.title("🏗️ Geotechnical Strength Predictor")
st.markdown("Predict **California Bearing Ratio (CBR)** and **Unconfined Compressive Strength (UCS)** using machine learning models.")

# Sidebar navigation for prediction modes
st.sidebar.header("🕹️ Prediction Settings")
prediction_mode = st.sidebar.radio(
    "Choose Prediction Target:",
    ["Predict CBR Only", "Predict Both (CBR -> UCS)", "Predict UCS (Using an existing CBR value)"]
)

if model_cbr and model_ucs and scaler:
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Additive Content (%)")
            fly_ash = st.number_input("Fly Ash (%)", min_value=0.0, max_value=100.0, step=0.1, value=0.0)
            lime = st.number_input("Lime (%)", min_value=0.0, max_value=100.0, step=0.1, value=0.0)
            cement = st.number_input("Cement (%)", min_value=0.0, max_value=100.0, step=0.1, value=0.0)
            
            # Show manual CBR input field only if the third option is selected
            if prediction_mode == "Predict UCS (Using an existing CBR value)":
                known_cbr = st.number_input("Enter known CBR (%)", min_value=0.0, max_value=100.0, step=0.1, value=5.0)
            else:
                known_cbr = None
            
        with col2:
            st.subheader("Physical Properties")
            ll = st.number_input("Liquid Limit LL (%)", min_value=0.0, max_value=200.0, step=0.1, value=35.0)
            pl = st.number_input("Plastic Limit PL (%)", min_value=0.0, max_value=100.0, step=0.1, value=20.0)
            omc = st.number_input("OMC (%)", min_value=0.0, max_value=50.0, step=0.1, value=12.0)
            mdd = st.number_input("MDD (g/cm³)", min_value=0.0, max_value=3.0, step=0.01, value=1.8)

        submit = st.form_submit_button("Run Prediction")

    if submit:
        # Dictionary to store the current record row
        record = {
            "Fly Ash (%)": fly_ash, "Lime (%)": lime, "Cement (%)": cement,
            "LL (%)": ll, "PL (%)": pl, "OMC (%)": omc, "MDD (g/cm³)": mdd,
            "Predicted CBR (%)": "N/A", "Predicted UCS": "N/A"
        }
        
        # Function to handle 9-feature MinMaxScaler matching requirement
        def scale_features(seven_features):
            # Pad array with 2 trailing zero dummy features to hit the expected 9 features
            dummy_pads = np.zeros((1, 2))
            nine_feature_row = np.hstack((seven_features, dummy_pads))
            scaled_row = scaler.transform(nine_feature_row)
            return scaled_row[:, :7]  # Extract and return only the 7 relevant scaled features

        # Core base features array
        base_features = np.array([[fly_ash, lime, cement, ll, pl, omc, mdd]])

        try:
            # MODE 1: Predict CBR Only
            if prediction_mode == "Predict CBR Only":
                scaled_base = scale_features(base_features)
                pred_cbr = model_cbr.predict(scaled_base)[0]
                
                record["Predicted CBR (%)"] = round(pred_cbr, 2)
                
                st.divider()
                st.metric(label="Predicted CBR (%)", value=f"{pred_cbr:.2f}%")

            # MODE 2: Chained Prediction (CBR first, then UCS)
            elif prediction_mode == "Predict Both (CBR -> UCS)":
                scaled_base = scale_features(base_features)
                pred_cbr = model_cbr.predict(scaled_base)[0]
                
                # Combine 7 base features + 1 predicted CBR feature to match UCS expectation
                # If your UCS model expects 8 total features:
                input_ucs = np.array([[fly_ash, lime, cement, ll, pl, omc, mdd, pred_cbr]])
                pred_ucs = model_ucs.predict(input_ucs)[0]
                
                record["Predicted CBR (%)"] = round(pred_cbr, 2)
                record["Predicted UCS"] = round(pred_ucs, 2)
                
                st.divider()
                res_col1, res_col2 = st.columns(2)
                res_col1.metric(label="Predicted CBR (%)", value=f"{pred_cbr:.2f}%")
                res_col2.metric(label="Predicted UCS", value=f"{pred_ucs:.2f}")

            # MODE 3: Predict UCS directly from a manual CBR value input
            elif prediction_mode == "Predict UCS (Using an existing CBR value)":
                input_ucs = np.array([[fly_ash, lime, cement, ll, pl, omc, mdd, known_cbr]])
                pred_ucs = model_ucs.predict(input_ucs)[0]
                
                record["Predicted CBR (%)"] = f"{known_cbr} (User Input)"
                record["Predicted UCS"] = round(pred_ucs, 2)
                
                st.divider()
                st.metric(label="Predicted UCS", value=f"{pred_ucs:.2f}")

            # Append the current run to history
            st.session_state.history.append(record)
            st.success("Calculations complete! Data appended to history table.")

        except ValueError as e:
            st.error(f"Shape error in processing arrays. Please check model feature counts. Detail: {e}")

    # --- History Tracking & CSV Download Block ---
    if st.session_state.history:
        st.subheader("📊 Session Log & History")
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)
        
        # Convert dataframe to CSV byte format
        csv_data = df_history.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download All History as CSV",
            data=csv_data,
            file_name="soil_predictions_log.csv",
            mime="text/csv"
        )
else:
    st.warning("Please make sure your asset models ('model_cbr.pkl', 'model_ucs.pkl', 'scaler.pkl') are placed in this folder.")