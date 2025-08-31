from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    )

from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split

# For preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
)

from sklearn.metrics import mean_absolute_error

import joblib
import pickle
import json







import streamlit as st
import pandas as pd
import joblib
from custom_transformers import (
    DataFrameImputer,
    OutlierThresholdTransformer,
    CustomOneHotEncoder,
    LogTransform,
    RobustScaleTransform,
    EnsemblePredictor,
    
    )

# ================================
# Load the trained pipeline
# ================================
@st.cache_resource
def load_pipeline():
    return joblib.load("final_pipeline.joblib")

pipeline = load_pipeline()

# ================================
# Define input columns
# ================================
numeric_columns = [
    'year', 'adult mortality', 'alcohol', 'hepatitis b', 'measles', 'bmi',
    'under-five deaths', 'polio', 'total expenditure', 'diphtheria',
    'hiv/aids', 'gdp', 'population', 'schooling', 'thinness'
]

cat_columns = ['country', 'status']

# ================================
# Streamlit App UI
# ================================
st.set_page_config(page_title="Life Expectancy Prediction", layout="centered")
st.title("🌍 Life Expectancy Prediction App")
st.write("Enter the details below to predict life expectancy.")

# Create input form
with st.form("prediction_form"):
    st.subheader("Enter Information:")

    # Numeric inputs
    numeric_inputs = {}
    for col in numeric_columns:
        numeric_inputs[col] = st.number_input(
            f"{col.capitalize()}",
            value=0.0 if col != "year" else 2000.0,
            step=1.0
        )

    # Categorical inputs
    country = st.text_input("Country")
    status = st.selectbox("Status", ["Developed", "Developing"])

    submitted = st.form_submit_button("Predict")

# ================================
# Make prediction
# ================================
if submitted:
    # Build input dataframe
    input_data = pd.DataFrame([{**numeric_inputs, "country": country, "status": status}])

    # Predict
    try:
        prediction = pipeline.predict(input_data)[0]
        st.success(f"✅ Predicted Life Expectancy: **{prediction:.2f} years**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
