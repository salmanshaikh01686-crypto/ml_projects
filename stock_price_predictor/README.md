# Stock Price Predictor

A Python + Streamlit machine-learning project that estimates the next trading session's closing price from historical stock data.

## Features
- Historical daily data from Yahoo Finance via `yfinance`
- Feature engineering: daily/5-day returns, moving averages, volatility and volume change
- Linear Regression and Random Forest models
- Chronological 80/20 train/test split (no random shuffling)
- Evaluation with MAE, RMSE and R²
- Interactive historical and actual-vs-predicted charts
- Next-session estimate for the selected ticker

## Requirements
- Python 3.10 or newer
- Internet connection when downloading stock data

## Setup (Windows)
Open Command Prompt in this folder and run:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open the dashboard in your browser. If it does not, open the local URL printed in the terminal (usually `http://localhost:8501`).

## Example tickers
- `RELIANCE.NS`
- `TCS.NS`
- `INFY.NS`
- `HDFCBANK.NS`
- `AAPL`

Use `.NS` for many NSE listings and `.BO` for many BSE listings.

## How the prediction is defined
For each trading session *t*, features are calculated using data available at the end of session *t*. The label is the adjusted closing price of session *t+1*. The final available session has no known label, so it is used only to produce the next-session estimate.

## Important limitations
- This is an educational demonstration, not financial advice or a trading signal.
- Historical patterns may not persist. News, earnings, liquidity, policy and macroeconomic events can dominate price movements.
- Yahoo Finance data availability and ticker coverage can vary.
- The test set is historical and cannot guarantee future performance.
- MAE/RMSE are in the stock's price units; compare models on the same ticker and period.

## Suggested future improvements
1. Add walk-forward validation rather than relying on one holdout period.
2. Compare against a naive baseline (next close = latest close).
3. Add transaction costs and backtest a defined trading strategy before discussing profitability.
4. Try XGBoost/LightGBM only after establishing a sound baseline.
5. Store downloaded data locally and log model versions for reproducibility.
