import streamlit as st
import yfinance as yf
import numpy as np

st.title("Quantitative Equity Hedge Fund Dashboard")

tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

data = yf.download(
    tickers,
    start="2020-01-01",
    auto_adjust=True
)["Close"]

returns = data.pct_change().dropna()

annual_return = returns.mean().mean() * 252
annual_volatility = returns.std().mean() * np.sqrt(252)
sharpe_ratio = annual_return / annual_volatility

col1, col2, col3 = st.columns(3)

col1.metric(
    "Annual Return",
    f"{annual_return:.2%}"
)

col2.metric(
    "Risk",
    f"{annual_volatility:.2%}"
)

col3.metric(
    "Sharpe Ratio",
    f"{sharpe_ratio:.2f}"
)

st.subheader("Stock Prices")
st.line_chart(data)

st.subheader("Daily Returns")
st.line_chart(returns)
