"""A simple form interface for the BTC/USD next-day Random Forest model."""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


DATA_FILE = Path(__file__).with_name("btc_usd_daily.csv")
FEATURES = ["close", "volume", "lag_1", "lag_7", "lag_14", "ma_7"]


@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_FILE)
    data["lag_1"] = data["close"].shift(1)
    data["lag_7"] = data["close"].shift(7)
    data["lag_14"] = data["close"].shift(14)
    data["ma_7"] = data["close"].rolling(7).mean()
    data["target_next_day"] = data["close"].shift(-1)
    data = data.dropna()

    split = int(len(data) * 0.8)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(data[FEATURES].iloc[:split], data["target_next_day"].iloc[:split])
    return model, data.iloc[-1]


st.set_page_config(page_title="Bitcoin Price Predictor", page_icon="B", layout="centered")
st.title("Bitcoin Price Predictor")
st.write("Enter market data, then get a next-day BTC/USD closing-price estimate.")

if not DATA_FILE.exists():
    st.error("Dataset not found. Run download_data.py first.")
    st.stop()

model, latest = train_model()
st.caption(f"Default values are from the newest dataset record: {latest['date']}.")

with st.form("prediction_form"):
    current_close = st.number_input("Current closing price (USD)", min_value=0.0, value=float(latest["close"]), step=100.0)
    current_volume = st.number_input("Current trading volume", min_value=0.0, value=float(latest["volume"]), step=100.0)
    yesterday_close = st.number_input("Yesterday's closing price (USD)", min_value=0.0, value=float(latest["lag_1"]), step=100.0)
    price_7_days_ago = st.number_input("Closing price 7 days ago (USD)", min_value=0.0, value=float(latest["lag_7"]), step=100.0)
    price_14_days_ago = st.number_input("Closing price 14 days ago (USD)", min_value=0.0, value=float(latest["lag_14"]), step=100.0)
    average_price = st.number_input("Average closing price over last 7 days (USD)", min_value=0.0, value=float(latest["ma_7"]), step=100.0)
    submitted = st.form_submit_button("Predict next-day price")

if submitted:
    question = pd.DataFrame([{
        "close": current_close,
        "volume": current_volume,
        "lag_1": yesterday_close,
        "lag_7": price_7_days_ago,
        "lag_14": price_14_days_ago,
        "ma_7": average_price,
    }])
    prediction = model.predict(question[FEATURES])[0]
    st.success(f"Estimated next-day BTC/USD closing price: ${prediction:,.2f}")
    st.caption("Educational model output only. Cryptocurrency prices are volatile and this is not trading advice.")
