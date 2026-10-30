import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

st.title("📈 Stock Price Predictor")
st.caption("Educational machine-learning demo using historical market data. Predictions are estimates, not investment advice.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock ticker", value="RELIANCE.NS",
                           help="Examples: RELIANCE.NS, TCS.NS, INFY.NS, AAPL")
    period = st.selectbox("Historical period", ["1y", "2y", "5y", "10y"], index=2)
    model_choice = st.selectbox("Model for next-day estimate",
                                ["Random Forest", "Linear Regression"])
    st.caption("For Indian NSE stocks, use the .NS suffix. For BSE, use .BO.")

@st.cache_data(ttl=900)
def download_data(symbol: str, history: str) -> pd.DataFrame:
    data = yf.download(symbol, period=history, interval="1d",
                       auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    # yfinance may return either flat or multi-index columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename_axis("Date").reset_index()
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not all(c in data.columns for c in required):
        return pd.DataFrame()
    return data[required].dropna().sort_values("Date").reset_index(drop=True)

def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Features available after a trading day closes; target is next session's close."""
    df = raw.copy()
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Volatility_10"] = df["Return_1d"].rolling(10).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Target_Next_Close"] = df["Close"].shift(-1)
    return df.replace([np.inf, -np.inf], np.nan)

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Return_1d", "Return_5d", "MA_5", "MA_10", "MA_20",
    "Volatility_10", "Volume_Change"
]

raw = download_data(ticker.strip().upper(), period)
if raw.empty:
    st.error("No data was returned. Check the ticker symbol and try again.")
    st.stop()

data = make_features(raw)
supervised = data.dropna(subset=FEATURES + ["Target_Next_Close"]).copy()

if len(supervised) < 80:
    st.error("Not enough usable historical data. Select a longer period (at least 1 year is recommended).")
    st.stop()

# Chronological split: never shuffle time-series observations.
split = int(len(supervised) * 0.8)
train = supervised.iloc[:split]
test = supervised.iloc[split:]

models = {
    "Linear Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
}

results = []
fitted = {}
for name, estimator in models.items():
    estimator.fit(train[FEATURES], train["Target_Next_Close"])
    predictions = estimator.predict(test[FEATURES])
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(test["Target_Next_Close"], predictions),
        "RMSE": np.sqrt(mean_squared_error(test["Target_Next_Close"], predictions)),
        "R²": r2_score(test["Target_Next_Close"], predictions)
    })
    fitted[name] = estimator

metrics = pd.DataFrame(results).set_index("Model")
selected = models[model_choice]

# Refit selected model on every row for which the next-session target is known.
selected.fit(supervised[FEATURES], supervised["Target_Next_Close"])
latest_features = data.dropna(subset=FEATURES).iloc[[-1]]
next_estimate = float(selected.predict(latest_features[FEATURES])[0])
last_close = float(raw["Close"].iloc[-1])
change = next_estimate - last_close
change_pct = (change / last_close) * 100 if last_close else 0

c1, c2, c3 = st.columns(3)
c1.metric("Latest available close", f"{last_close:,.2f}")
c2.metric("Estimated next-session close", f"{next_estimate:,.2f}",
          f"{change_pct:+.2f}% vs latest close")
c3.metric("Last data date", pd.to_datetime(raw["Date"].iloc[-1]).strftime("%d %b %Y"))

st.subheader("Historical closing price")
price_fig = go.Figure()
price_fig.add_trace(go.Scatter(x=raw["Date"], y=raw["Close"],
                               mode="lines", name="Adjusted close"))
price_fig.update_layout(xaxis_title="Date", yaxis_title="Price",
                        hovermode="x unified", height=420)
st.plotly_chart(price_fig, use_container_width=True)

st.subheader("Model evaluation (chronological 80/20 split)")
st.dataframe(metrics.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"}),
             use_container_width=True)

test_predictions = fitted[model_choice].predict(test[FEATURES])
comparison = pd.DataFrame({
    "Date": test["Date"],
    "Actual next close": test["Target_Next_Close"],
    "Predicted next close": test_predictions
})
fig = go.Figure()
fig.add_trace(go.Scatter(x=comparison["Date"], y=comparison["Actual next close"],
                         name="Actual", mode="lines"))
fig.add_trace(go.Scatter(x=comparison["Date"], y=comparison["Predicted next close"],
                         name="Predicted", mode="lines"))
fig.update_layout(title=f"{model_choice}: test-set predictions",
                  xaxis_title="Feature date", yaxis_title="Next-session close",
                  hovermode="x unified", height=420)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "Important: This is a learning project, not a dependable trading system. "
    "Stock prices are affected by news, earnings, macroeconomic events and market sentiment. "
    "A low historical error does not guarantee future accuracy. The estimate is for the next "
    "trading session after the latest data returned by Yahoo Finance."
)

with st.expander("How this project works"):
    st.markdown("""
    1. Downloads daily adjusted OHLCV data with `yfinance`.
    2. Creates lagged/rolling indicators using information available at the end of each session.
    3. Uses the next session's closing price as the target.
    4. Splits observations chronologically (80% train, 20% test); it does not shuffle dates.
    5. Compares Linear Regression and Random Forest using MAE, RMSE and R².
    6. Retrains the selected model on all labelled rows and estimates the next session's close.
    """)
