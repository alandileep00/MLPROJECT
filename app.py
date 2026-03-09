import streamlit as st
import pandas as pd
import joblib

# Set page config for aesthetics
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the pre-trained models from disk."""
    mlr_model = joblib.load('mlr_model.pkl')
    slr_model = joblib.load('slr_model.pkl')
    pr_model = joblib.load('pr_model.pkl')
    poly_converter = joblib.load('poly_converter.pkl')
    log_reg_model = joblib.load('log_reg_model.pkl')
    log_reg_threshold = joblib.load('log_reg_threshold.pkl')
    knn_model = joblib.load('knn_model.pkl')
    knn_scaler = joblib.load('knn_scaler.pkl')
    return mlr_model, slr_model, pr_model, poly_converter, log_reg_model, log_reg_threshold, knn_model, knn_scaler

try:
    mlr_model, slr_model, pr_model, poly_converter, log_reg_model, log_reg_threshold, knn_model, knn_scaler = load_models()
except Exception as e:
    st.error(f"Error loading models. Please ensure 'train_models.py' has been run. Details: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.title("🚗 Navigation")
st.sidebar.markdown('---')
app_mode = st.sidebar.radio(
    "Select Prediction Model",
    ["Multiple Linear Regression", "Simple Linear Regression", "Polynomial Regression", "K-Nearest Neighbors", "Logistic Regression (Classification)"]
)
st.sidebar.markdown('---')


# --- Multiple Linear Regression View ---
if app_mode == "Multiple Linear Regression":
    st.title("Price Predictor: Multiple Linear Regression")
    st.markdown("Predict the selling price based on *multiple* features.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Car Details")
        year = st.number_input("Year of Manufacture", min_value=1990, max_value=2024, value=2015, step=1)
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000, step=1000)
        seats = st.number_input("Number of Seats", min_value=2, max_value=14, value=5, step=1)
        
    with col2:
        st.subheader("Engine Specifications")
        mileage = st.number_input("Mileage (km/ltr/kg)", min_value=0.0, max_value=50.0, value=20.0, step=0.5)
        engine = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, value=1197, step=100)
        max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=600.0, value=74.0, step=1.0)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Price (MLR)", key="btn_mlr"):
        # Make prediction
        input_data = pd.DataFrame({
            'year': [year],
            'km_driven': [km_driven],
            'mileage(km/ltr/kg)': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'seats': [seats]
        })
        
        try:
            prediction = mlr_model.predict(input_data)[0]
            
            st.markdown(f"""
            <div class="metric-card">
                <h2>Predicted Selling Price</h2>
                <h1 style="color: #27ae60; font-size: 3rem;">₹ {prediction:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            if prediction < 0:
                st.warning("The model predicted a negative value, which usually means the input parameters are far outside the normal dataset boundaries.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- Simple Linear Regression View ---
elif app_mode == "Simple Linear Regression":
    st.title("Price Predictor: Simple Linear Regression")
    st.markdown("Predict the selling price based *only* on Kilometers Driven, using a simple linear relationship.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_center, _ = st.columns([1, 1])
    with col_center:
        km_driven_slr = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000, step=1000, key="km_slr")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Predict Price (SLR)", key="btn_slr"):
            input_data_slr = pd.DataFrame({'km_driven': [km_driven_slr]})
            
            try:
                prediction_slr = slr_model.predict(input_data_slr)[0]
                
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 10px;">
                    <h2>Predicted Selling Price</h2>
                    <h1 style="color: #2980b9; font-size: 3rem;">₹ {prediction_slr:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                if prediction_slr < 0:
                    st.warning("The model predicted a negative value, which usually means the input parameters are far outside the normal dataset boundaries.")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# --- Polynomial Regression View ---
elif app_mode == "Polynomial Regression":
    st.title("Price Predictor: Polynomial Regression")
    st.markdown("Predict the selling price based *only* on Kilometers Driven, using a polynomial degree (2) relationship.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_center, _ = st.columns([1, 1]) # Keep input constrained
    with col_center:
        km_driven_poly = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000, step=1000, key="km_poly")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Predict Price (Polynomial)", key="btn_poly"):
            input_data_poly = pd.DataFrame({'km_driven': [km_driven_poly]})
            
            try:
                # Transform input using polynomial converter
                X_poly = poly_converter.transform(input_data_poly)
                prediction_poly = pr_model.predict(X_poly)[0]
                
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 10px;">
                    <h2>Predicted Selling Price</h2>
                    <h1 style="color: #8e44ad; font-size: 3rem;">₹ {prediction_poly:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# --- K-Nearest Neighbors View ---
elif app_mode == "K-Nearest Neighbors":
    st.title("Price Predictor: K-Nearest Neighbors")
    st.markdown("Predict the selling price using the **K-Nearest Neighbors (KNN)** algorithm.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Car Details")
        year_knn = st.number_input("Year of Manufacture", min_value=1990, max_value=2024, value=2015, step=1, key="y_knn")
        km_driven_knn = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000, step=1000, key="km_knn")
        seats_knn = st.number_input("Number of Seats", min_value=2, max_value=14, value=5, step=1, key="s_knn")
        
    with col2:
        st.subheader("Engine Specifications")
        mileage_knn = st.number_input("Mileage (km/ltr/kg)", min_value=0.0, max_value=50.0, value=20.0, step=0.5, key="m_knn")
        engine_knn = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, value=1197, step=100, key="e_knn")
        max_power_knn = st.number_input("Max Power (bhp)", min_value=20.0, max_value=600.0, value=74.0, step=1.0, key="mp_knn")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Price (KNN)", key="btn_knn"):
        input_data_knn = pd.DataFrame({
            'year': [year_knn],
            'km_driven': [km_driven_knn],
            'mileage(km/ltr/kg)': [mileage_knn],
            'engine': [engine_knn],
            'max_power': [max_power_knn],
            'seats': [seats_knn]
        })
        
        try:
            # Scale the input data using the saved scaler
            scaled_input = knn_scaler.transform(input_data_knn)
            
            # Make prediction
            prediction_knn = knn_model.predict(scaled_input)[0]
            
            st.markdown(f"""
            <div class="metric-card">
                <h2>Predicted Selling Price</h2>
                <h1 style="color: #e74c3c; font-size: 3rem;">₹ {prediction_knn:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- Logistic Regression View ---
elif app_mode == "Logistic Regression (Classification)":
    st.title("Price Predictor: Logistic Regression")
    st.markdown(f"Classify if the car is a **High Value** (above ₹ {log_reg_threshold:,.2f}) or **Standard Value** car.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Car Details")
        year_lr = st.number_input("Year of Manufacture", min_value=1990, max_value=2024, value=2015, step=1, key="y_lr")
        km_driven_lr = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000, step=1000, key="km_lr")
        seats_lr = st.number_input("Number of Seats", min_value=2, max_value=14, value=5, step=1, key="s_lr")
        
    with col2:
        st.subheader("Engine Specifications")
        mileage_lr = st.number_input("Mileage (km/ltr/kg)", min_value=0.0, max_value=50.0, value=20.0, step=0.5, key="m_lr")
        engine_lr = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, value=1197, step=100, key="e_lr")
        max_power_lr = st.number_input("Max Power (bhp)", min_value=20.0, max_value=600.0, value=74.0, step=1.0, key="mp_lr")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Classify Car Value", key="btn_lr"):
        input_data_lr = pd.DataFrame({
            'year': [year_lr],
            'km_driven': [km_driven_lr],
            'mileage(km/ltr/kg)': [mileage_lr],
            'engine': [engine_lr],
            'max_power': [max_power_lr],
            'seats': [seats_lr]
        })
        
        try:
            prediction_lr = log_reg_model.predict(input_data_lr)[0]
            probability = log_reg_model.predict_proba(input_data_lr)[0]
            
            if prediction_lr == 1:
                class_label = "High Value Car"
                color = "#27ae60" # Green
                prob_text = f"Probability: {probability[1]*100:.1f}%"
            else:
                class_label = "Standard Value Car"
                color = "#f39c12" # Orange
                prob_text = f"Probability: {probability[0]*100:.1f}%"
            
            st.markdown(f"""
            <div class="metric-card">
                <h2>Classification Result</h2>
                <h1 style="color: {color}; font-size: 3rem;">{class_label}</h1>
                <p style="font-size: 1.2rem; color: #7f8c8d;">{prob_text}</p>
                <p>Based on a median threshold of ₹ {log_reg_threshold:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Classification Error: {e}")
