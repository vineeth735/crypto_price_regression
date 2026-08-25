# Cryptocurrency Price Prediction: Regression Starter

This folder creates a larger, clean dataset for a beginner-friendly regression project.

It downloads approximately 11 years of daily BTC/USD market candles (about 2,900 rows). Each row has the date, open, high, low, close, and trading volume. The source is Coinbase Exchange's public market-data API.

## Run it

From this folder, run:

```bash
python3 download_data.py
python3 train_baseline.py
```

The download creates `btc_usd_daily.csv`. The model uses the previous day's close, the price 7 and 14 days ago, and yesterday's volume to predict the next daily closing price.

It uses the first 80% of the timeline for training and the newest 20% for testing. This is important: financial data must stay in date order, otherwise the model accidentally learns from the future.

## Use another cryptocurrency

For Ethereum, run:

```bash
python3 download_data.py --product ETH-USD --years 11 --output eth_usd_daily.csv
python3 train_baseline.py --data eth_usd_daily.csv
```

Examples of larger data sizes:

| Period | Approximate daily rows |
| --- | ---: |
| 1 year | 365 |
| 3 years | 1,095 |
| 5 years | 1,825 |
| 8 years | 2,920 |
| 11 years | 4,015|

For hourly or minute-by-minute prediction, download a separate high-frequency dataset. Start with daily data first: it is easier to clean, explain, and evaluate for a college project.

## Use the no-code prediction form

Install the app dependencies once, then start the app:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

The browser page lets you enter the market values and click **Predict next-day price**. No Python needs to be typed after the app opens.

## Dataset source

Coinbase documents its public candles endpoint here: https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductcandles

This dataset is for educational analysis, not trading advice.
