import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import os

# Set page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        color: #1a1a1a;
    }
    .css-1d391kg {
        padding-top: 0rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
    }
    .prediction-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        color: #1a1a1a;
    }
    .model-metrics {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #1a1a1a;
    }
    .title-container {
        background-color: white;
        padding: 2rem;
        border-radius: 0 0 1rem 1rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: #1a1a1a;
    }
    .subtitle {
        color: #666;
        font-size: 1.2rem;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #666;
    }
    [data-testid="stMetricValue"] {
        color: #ff4b4b !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricLabel"] {
        color: #1a1a1a !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Streamlit Select box styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        color: #1a1a1a !important;
    }
    .stSelectbox > div > div:hover {
        border-color: #ff4b4b;
    }
    .stSelectbox label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }

    /* Date input styling */
    .stDateInput > div > div {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        color: #1a1a1a !important;
    }
    .stDateInput > div > div:hover {
        border-color: #ff4b4b;
    }
    .stDateInput label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }

    /* Time input styling */
    .stTimeInput > div > div {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        color: #1a1a1a !important;
    }
    .stTimeInput > div > div:hover {
        border-color: #ff4b4b;
    }
    .stTimeInput label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }

    /* Number input styling */
    .stNumberInput > div > div {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        color: #1a1a1a !important;
    }
    .stNumberInput > div > div:hover {
        border-color: #ff4b4b;
    }
    .stNumberInput label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }

    /* Input labels general styling */
    .css-81oif8 {
        font-size: 0.9rem !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }

    /* Dataframe styling */
    .dataframe {
        font-size: 0.9rem !important;
    }
    .dataframe tbody tr:nth-child(odd) {
        background-color: #f8f9fa;
    }
    .dataframe tbody tr:hover {
        background-color: #f0f0f0;
    }
    .dataframe th {
        background-color: #f8f9fa;
        font-weight: 600;
        color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)


def load_model(model_name):
    """Load the selected model from pickle file."""
    model_path = os.path.join('models', f'{model_name}.pkl')
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    return data['model'], data['columns'], data['metrics']

def load_model_comparison():
    """Load the model comparison metrics."""
    comparison_path = os.path.join('models', 'model_comparison.csv')
    if os.path.exists(comparison_path):
        return pd.read_csv(comparison_path, index_col=0)
    return None

def predict_price(model, columns, features):
    """Make prediction using the trained model."""
    input_df = pd.DataFrame(0, index=[0], columns=columns)
    for key, value in features.items():
        if key in input_df.columns:
            input_df[key] = value
        elif key.startswith(('Airline_', 'Source_', 'Destination_')):
            if key in input_df.columns:
                input_df[key] = 1
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)

def main():
    # Title with custom styling
    st.markdown("""
        <div class="title-container">
            <h1>✈️ Flight Price Predictor</h1>
            <p class="subtitle">Predict your flight prices using machine learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    model_names = {
        'random_forest': '🌳 Random Forest',
        'gradient_boosting': '🚀 Gradient Boosting',
        'extra_trees': '🌲 Extra Trees',
        'linear_regression': '📈 Linear Regression'
    }
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_model = st.selectbox(
            'Select Your Prediction Model',
            list(model_names.keys()),
            format_func=lambda x: model_names[x]
        )
    
    try:
        # Load model and display metrics
        model, columns, metrics = load_model(selected_model)
        
        st.markdown("""
            <div class="model-metrics">
                <h3>🎯 Model Performance Metrics</h3>
            </div>
        """, unsafe_allow_html=True)
        
        metric1, metric2, metric3 = st.columns(3)
        with metric1:
            st.metric("Training R² Score", f"{metrics['train_score']:.3f}")
        with metric2:
            st.metric("Testing R² Score", f"{metrics['test_score']:.3f}")
        with metric3:
            st.metric("Mean Absolute Error", f"₹{metrics['mae']:.2f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Create input fields with better organization
        st.markdown("""
            <div class="prediction-card">
                <h3>🛫 Flight Details</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### From")
            sources = ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']
            source = st.selectbox('Select Source City', sources, key='source')
            
            st.markdown("##### Departure")
            dep_date = st.date_input("Date", min_value=datetime.now(), key='dep_date')
            dep_time = st.time_input("Time", key='dep_time')
            
            st.markdown("##### Stops")
            total_stops = st.number_input("Number of Stops", 
                                        min_value=0, max_value=4, 
                                        step=1, key='stops')
        
        with col2:
            st.markdown("##### To")
            destinations = ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata']
            destination = st.selectbox('Select Destination City', destinations, key='dest')
            
            st.markdown("##### Arrival")
            arrival_date = st.date_input("Date", min_value=dep_date, key='arr_date')
            arrival_time = st.time_input("Time", key='arr_time')
            
            st.markdown("##### Airline")
            airlines = [
                'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
                'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet',
                'Trujet', 'Vistara', 'Vistara Premium economy'
            ]
            airline = st.selectbox('Select Airline', airlines, key='airline')
        
        # Center the predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button('Predict Price')
        
        if predict_button:
            if source == destination:
                st.error("🚫 Source and Destination cannot be the same!")
                return
            
            dep_datetime = datetime.combine(dep_date, dep_time)
            arrival_datetime = datetime.combine(arrival_date, arrival_time)
            
            if arrival_datetime <= dep_datetime:
                st.error("🚫 Arrival time must be after departure time!")
                return
            
            duration = arrival_datetime - dep_datetime
            duration_hours = duration.days * 24 + duration.seconds // 3600
            duration_mins = (duration.seconds % 3600) // 60
            
            features = {
                'Journey_Day': dep_date.day,
                'Journey_Month': dep_date.month,
                'Dep_Hour': dep_time.hour,
                'Dep_Min': dep_time.minute,
                'Arrival_Hour': arrival_time.hour,
                'Arrival_Min': arrival_time.minute,
                'Duration_Hours': duration_hours,
                'Duration_Mins': duration_mins,
                'Total_Stops': total_stops,
                f'Airline_{airline}': 1,
                f'Source_{source}': 1,
                f'Destination_{destination}': 1
            }
            
            price = predict_price(model, columns, features)
            
            # Display prediction in a nice card
            st.markdown("""
                <div class="prediction-card">
                    <h3>🎯 Price Prediction</h3>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                    <div style='text-align: center; background-color: #f8f9fa; 
                              padding: 2rem; border-radius: 1rem; margin: 1rem 0;'>
                        <h2 style='color: #ff4b4b; margin-bottom: 0.5rem;'>
                            ₹{price:,.2f}
                        </h2>
                        <p style='color: #666; margin: 0;'>Predicted Price</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Display flight details
            st.markdown("""
                <div class="prediction-card">
                    <h3>✈️ Flight Summary</h3>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    - 🛫 **From:** {source}
                    - 🛬 **To:** {destination}
                    - ⏱️ **Duration:** {duration_hours}h {duration_mins}m
                """)
            with col2:
                st.markdown(f"""
                    - 🛑 **Stops:** {total_stops}
                    - ✈️ **Airline:** {airline}
                    - 🤖 **Model:** {model_names[selected_model]}
                """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the model files exist in the 'models' directory and all inputs are valid.")
    
    # Display model comparison if available
    try:
        comparison_df = load_model_comparison()
        if comparison_df is not None:
            st.markdown("""
                <div class="prediction-card">
                    <h3>📊 Model Comparison</h3>
                </div>
            """, unsafe_allow_html=True)
            st.dataframe(comparison_df.style.highlight_max(axis=1))
    except Exception as e:
        st.warning("Could not load model comparison data.")

if __name__ == "__main__":
    main()