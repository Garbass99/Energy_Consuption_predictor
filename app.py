import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# Load the saved model
@st.cache_resource
def load_model():
    with open('Heartn.pkl', 'rb') as f:
        model = joblib.load(f)
    return model

model = load_model()

# Title and description
st.title("Heart Attack Risk Prediction")
st.write("""
This app predicts the likelihood of heart attack based on health parameters.
Adjust the values in the sidebar and see the prediction update in real-time.
""")

# Sidebar for user input
st.sidebar.header('Patient Health Parameters')

def get_user_input():
    """Create input widgets and return as DataFrame"""
    # Numerical inputs
    age = st.sidebar.slider('Age', 20, 100, 50)
    trtbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 600, 250)
    thalachh = st.sidebar.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
    caa = st.sidebar.slider('Number of Major Vessels', 0, 4, 0)
    
    # Categorical inputs
    sex = st.sidebar.radio('Sex', ['Female', 'Male'])
    cp = st.sidebar.selectbox('Chest Pain Type', 
                            ['Typical Angina (0)', 
                             'Atypical Angina (1)', 
                             'Non-anginal Pain (2)', 
                             'Asymptomatic (3)'])
    fbs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dl', ['No (0)', 'Yes (1)'])
    restecg = st.sidebar.selectbox('Resting ECG Results', 
                                 ['Normal (0)', 
                                  'ST-T Wave Abnormality (1)', 
                                  'Probable LV Hypertrophy (2)'])
    exng = st.sidebar.radio('Exercise Induced Angina', ['No (0)', 'Yes (1)'])
    slp = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', 
                             ['Upsloping (0)', 'Flat (1)', 'Downsloping (2)'])
    thall = st.sidebar.selectbox('Thalium Stress Test Result', 
                               ['Normal (0)', 'Fixed Defect (1)', 'Reversible Defect (2)'])
    
    # Convert categorical inputs to numerical
    sex = 1 if sex == 'Male' else 0
    cp = int(cp.split('(')[1].replace(')', ''))
    fbs = int(fbs.split('(')[1].replace(')', ''))
    restecg = int(restecg.split('(')[1].replace(')', ''))
    exng = int(exng.split('(')[1].replace(')', ''))
    slp = int(slp.split('(')[1].replace(')', ''))
    thall = int(thall.split('(')[1].replace(')', ''))
    
    # Create DataFrame
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trtbps': trtbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalachh': thalachh,
        'exng': exng,
        'oldpeak': oldpeak,
        'slp': slp,
        'caa': caa,
        'thall': thall
    }
    
    return pd.DataFrame(user_data, index=[0])

# Get user input
user_input = get_user_input()

# Display user input
st.subheader('Patient Input Parameters')
st.write(user_input)

# Make prediction
try:
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)
    
    # Display prediction
    st.subheader('Prediction Result')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Risk Level", 
                 value="High Risk ⚠️" if prediction[0] == 1 else "Low Risk ❤️")
    
    with col2:
        risk_percent = round(prediction_proba[0][1] * 100, 1)
        st.metric("Risk Probability", 
                 value=f"{risk_percent}%")
    
    # Visual gauge
    st.progress(int(risk_percent))
    
    # Probability breakdown
    st.subheader('Probability Breakdown')
    prob_df = pd.DataFrame({
        'Class': ['No Heart Disease', 'Heart Disease Risk'],
        'Probability': prediction_proba[0]
    })
    st.bar_chart(prob_df.set_index('Class'))
    
except Exception as e:
    st.error(f"Error making prediction: {str(e)}")

# Add some info about features if checkbox is checked
if st.checkbox("Show feature descriptions"):
    st.subheader("Feature Descriptions")
    feature_info = """
    - **age**: Age in years
    - **sex**: 1 = male, 0 = female  
    - **cp**: Chest pain type (0-3)
    - **trtbps**: Resting blood pressure (mm Hg)
    - **chol**: Serum cholesterol (mg/dl)
    - **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    - **restecg**: Resting electrocardiographic results (0-2)
    - **thalachh**: Maximum heart rate achieved
    - **exng**: Exercise induced angina (1 = yes; 0 = no)
    - **oldpeak**: ST depression induced by exercise relative to rest
    - **slp**: Slope of the peak exercise ST segment (0-2)
    - **caa**: Number of major vessels (0-4) colored by fluoroscopy
    - **thall**: Thalium stress test result (0-3)
    """
    st.markdown(feature_info)

# Footer
st.markdown("---")
st.caption("""
Note: This prediction is for informational purposes only and should not replace professional medical advice.
""")

