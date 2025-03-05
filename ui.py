import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pycaret.regression import predict_model

# Load trained model
model = joblib.load("best_stock_model.pkl")

# Cache the function correctly
@st.cache_data
def load_data():
    df = pd.read_csv("processed_stocks_data.csv")  # Load your preprocessed dataset
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = pd.get_dummies(df, columns=['Stock'])  # One-hot encoding for Stock column
    df.drop(columns=['Date'], inplace=True)  # Drop Date column as in preprocessing
    return df

# ✅ Load data once
df = load_data()

# Select Features
stock_columns = ["Stock_AAPL", "Stock_AMZN", "Stock_GOOGL", "Stock_MSFT", "Stock_TSLA"]
input_data = df[["Open", "High", "Low", "Close", "Volume", "Year", "Month", "Day"] + stock_columns]

# Predict stock returns
def predict_returns(data):
    predicted_data = predict_model(model, data=data)
    stock_names = [col.replace("Stock_", "") for col in stock_columns]  # Extract readable stock names
    return dict(zip(stock_names, predicted_data["prediction_label"].values[:len(stock_names)]))

# Allocate investments using MPT-based strategy
def allocate_investment(predicted_returns, investment_amount, diversify=True):
    sorted_returns = sorted(predicted_returns.items(), key=lambda x: x[1], reverse=True)
    
    if diversify:
        top_stocks = sorted_returns[:3]  # Select top 3
        total_return = sum([r[1] for r in top_stocks])
        allocations = {s: (r / total_return) * investment_amount for s, r in top_stocks}
    else:
        top_stock = sorted_returns[0]
        allocations = {top_stock[0]: investment_amount}
    
    return allocations

# Streamlit UI
st.title("📈 Smart Portfolio Optimizer")

investment_amount = st.number_input("💰 Investment Amount ($)", min_value=1000, value=10000, step=500)

diversify = st.checkbox("Diversify Portfolio?", value=True)

if st.button("Optimize Portfolio"):
    with st.spinner("Processing historical data & predicting returns..."):
        df = load_data()
        
        # Ensure the data format matches what the model expects
        input_data = df[["Open", "High", "Low", "Close", "Volume", "Year", "Month", "Day"] + stock_columns]
        predicted_returns = predict_returns(input_data)
        
        investment_strategy = allocate_investment(predicted_returns, investment_amount, diversify)
        
        st.subheader("💡 Recommended Investment Strategy")
        st.json(investment_strategy)