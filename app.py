import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(
    page_title="Cotton Price Prediction",
    layout="wide"
)

st.title("📈 Cotton Price Prediction Dashboard")
st.markdown("Final Model: **Linear Regression**")

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("US Cotton #2 Futures Historical Data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

df = load_data()

# ---------------------------------------------------
# Feature engineering
# --------------------
