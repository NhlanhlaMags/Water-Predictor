import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from custom_transformer import WaterFeatureEngineer
from sklearn.base import BaseEstimator, TransformerMixin
import os
from pathlib import Path

# ---------------------------
# Page Config
# ---------------------------
st.markdown('<div class ="header-banner"> 💧 Water Potability Predictor</div>', unsafe_allow_html=True)
# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #e0f7fa, #ffffff);
}
.header-banner {
    background: linear-gradient(90deg, #2196f3, #00bcd4);
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.card {
    padding: 15px;
    border-radius: 12px;
    background-color: #f0f4f8;
    margin-bottom: 15px;
    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
# ---------------------------
# Header + About Section
# ---------------------------

st.markdown("""
### 👥 Who We Are
**We are NaN Masters** — a curious crew of data explorers, problem solvers, and change-makers.
Our mission? To turn raw data into real-world impact.

We dive deep into numbers, patterns, and possibilities to help communities access **clean, safe, and sustainable water**.
For us, every dataset tells a story — and we use technology to make that story count.

With curiosity as our compass and innovation as our toolkit,
we’re here to prove that even from **NaN (Not a Number)** beginnings, great solutions can flow.
""")

# ---------------------------
# Problem Overview
# ---------------------------
st.markdown("---")
st.subheader("🌍 Why This Matters")
st.markdown("""
Many communities worldwide lack access to **safe drinking water**, and testing is often **slow or unavailable**.
Our model predicts whether water is safe or unsafe to drink using **chemical properties like pH, turbidity, hardness, and more**.

This helps communities and municipalities **take quick, data-driven action** to ensure water safety.
""")

class WaterFeatureEngineer(BaseEstimator, TransformerMixin):
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'updated_model.pkl')
    
    def __init__ (self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        st.warning(" Using fallback WaterFeatureEngineer - no transformations applied")
        return X.copy()

# ---------------------------
# Prediction Mode
# ---------------------------
@st.cache_resource
def load_model():
    try:
        # Get script directory
        current_dir = Path(__file__).parent

        # Use correct relative path
        model_path = current_dir / "Random_pipeline.pkl"

        # Debug output
        st.write(f"Loading model from: {model_path}")
        st.write(f"File exists: {os.path.exists(model_path)}")

        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            return None

        # Corrected syntax here
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
            st.success("✅ Model loaded successfully!")
            return {"model": model, "preprocessor": None} # Return model in a dictionary

    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model_info = load_model() # Call load_model once here

st.markdown("---")
st.subheader("🚀 Choose a Prediction Mode")
mode = st.radio(
    "Select how you'd like to predict water safety:",
    ("🔹 Manual Input", "📂 Batch CSV Upload"),
    horizontal=True
)
# ---------------------------
# Manual Input Mode
# ---------------------------
if mode == "🔹 Manual Input":
    st.sidebar.header("Enter Water Quality Features")

    def user_input_features():
        pH = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
        Hardness = st.sidebar.slider("Hardness (mg/L)", 0.0, 500.0, 150.0)
        Solids = st.sidebar.number_input("Solids (ppm)", 0.0, 50000.0, 20000.0)
        Chloramines = st.sidebar.slider("Chloramines (mg/L)", 0.0, 10.0, 3.0)
        Sulfate = st.sidebar.number_input("Sulfate (mg/L)", 0.0, 500.0, 200.0)
        Conductivity = st.sidebar.number_input("Conductivity (μS/cm)", 0.0, 1000.0, 400.0)
        Organic_carbon = st.sidebar.slider("Organic Carbon (mg/L)", 0.0, 30.0, 10.0)
        Trihalomethanes = st.sidebar.number_input("Trihalomethanes (µg/L)", 0.0, 150.0, 50.0)
        Turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 10.0, 4.0)

        data = {
            'ph': pH,
            'Hardness': Hardness,
            'Solids': Solids,
            'Chloramines': Chloramines,
            'Sulfate': Sulfate,
            'Conductivity': Conductivity,
            'Organic_carbon': Organic_carbon,
            'Trihalomethanes': Trihalomethanes,
            'Turbidity': Turbidity
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()
    st.subheader("🔍 Entered Water Quality Data:")
    st.write(input_df)

    if st.button("💧 Predict Water Safety"):
        if model_info is not None and model_info["model"] is not None:
            try:
                model = model_info["model"]
                prediction = model.predict(input_df)
                # Ensure predict_proba returns a 2D array
                prob = model.predict_proba(input_df)
                prob = prob[0] # Access the first (and only) prediction's probabilities
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                # Keep the Streamlit app running but show error
                prediction = None
                prob = None


            if prediction is not None:
                color = "green" if prediction[0] == 1 else "red"
                label = "✅ SAFE" if prediction[0] == 1 else "⚠️ UNSAFE"
                st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
                if prob is not None:
                     st.write(f"Probability: Unsafe: {round(prob[0]*100, 2)}% | Safe: {round(prob[1]*100, 2)}%")
        else:
            st.error("Model not loaded. Cannot make prediction.")

# ---------------------------
# Batch CSV Upload Mode
# ---------------------------
elif mode == "📂 Batch CSV Upload":
    st.subheader("📁 Upload a CSV File for Batch Predictions")
    st.markdown("""
    Upload a CSV file with these columns:
    `ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity`
    """)

    uploaded_file = st.file_uploader("Upload your water quality dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("🚀 Predict for All Rows"):
            if model_info is not None and model_info["model"] is not None:
                model = model_info["model"]
                try:
                    # Apply the pipeline to the uploaded DataFrame
                    # Assuming the pipeline handles feature engineering and scaling internally
                    df['Potability_Prediction'] = model.predict(df)
                    st.success("✅ Predictions generated successfully!")
                    st.dataframe(df.head())

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")
            else:
                st.error("Model not loaded. Cannot make batch predictions.")
    else:
        st.info("Please upload a CSV file to continue.")

# ---------------------------
# Model Insights Section
# ---------------------------
st.markdown("---")
st.subheader("📊 Model Insights")

# Demo metrics (replace with real model metrics)
accuracy = 0.85
precision = 0.82
recall = 0.80
f1 = 0.81

st.markdown(f"""
**Model Performance (Demo Metrics):**
- Accuracy: {accuracy}
- Precision: {precision}
- Recall: {recall}
- F1 Score: {f1}
""")

# Feature importance using Streamlit only (no matplotlib)
st.markdown("**Feature Importance:**")

# You'll need to get the actual feature importances from your trained model
# For Random Forest (if that's your final model), you can access model.feature_importances_
# Make sure the feature names match the order of features used during training
# For now, using placeholder data
feature_data = pd.DataFrame({
    'Feature': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
    'Importance': [0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04]
})

# Display as bar chart using Streamlit
st.bar_chart(feature_data.set_index('Feature'))

# Or display as table
st.dataframe(feature_data.sort_values('Importance', ascending=False))

# ---------------------------
# Meet the Team Section
# ---------------------------
st.markdown("---")
st.subheader("👩‍💻 Meet the Team")
st.markdown("""
1. **Snenhlanhla Nsele** - Data Scientist - [LinkedIn](https://www.linkedin.com/in/sinenhlanhla-nsele-126a6a18a)
2. **Nonhlanhla Magagula** - Data Scientist - [LinkedIn](https://www.linkedin.com/in/nonhlanhla-magagula-b741b3207)
3. **Thandiwe Mkhabela** - Data Scientist - [LinkedIn](https://www.linkedin.com/in/thandiwe-m)
4. **Thabiso Seema** - Software Engineer - [LinkedIn](https://www.linkedin.com/in/thabisoseema)
""")

# ---------------------------
# Model Versioning Info
# ---------------------------
st.markdown("---")
st.subheader("🧩 Model Versioning Info")
st.markdown("""
- **Model Version:** 1.0
- **Last Updated:** Nov 2025
- **Training Data:** 3,000+ samples
""")

# ---------------------------
# Footer
# ---------------------------
st.caption("Created with ❤️ by NaN Masters | Powered by Streamlit")
