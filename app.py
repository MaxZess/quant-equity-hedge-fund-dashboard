import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ====================================
# Titel
# ====================================

st.set_page_config(
    page_title="Quant Hedge Fund Dashboard",
    layout="wide"
)

st.title("Quantitative Equity Hedge Fund Dashboard")

# ====================================
# Aktien
# ====================================

tickers = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL"
]

# ====================================
# Daten laden
# ====================================

data = yf.download(
    tickers,
    start="2020-01-01",
    auto_adjust=True
)["Close"]

returns = data.pct_change().dropna()

# ====================================
# Portfolio-Kennzahlen
# ====================================

portfolio_returns = returns.mean(axis=1)

annual_return = portfolio_returns.mean() * 252

annual_volatility = (
    portfolio_returns.std()
    * np.sqrt(252)
)

sharpe_ratio = annual_return / annual_volatility

# ====================================
# Dashboard Metrics
# ====================================

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

# ====================================
# Kursverlauf
# ====================================

st.subheader("Stock Prices")

st.line_chart(data)

# ====================================
# Renditen
# ====================================

st.subheader("Daily Returns")

st.line_chart(returns)

# ====================================
# Portfolio Statistik
# ====================================

st.subheader("Portfolio Statistics")

weights = np.array([
    0.2,
    0.2,
    0.2,
    0.2,
    0.2
])

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

portfolio_return = np.sum(
    weights * mean_returns
)

portfolio_risk = np.sqrt(
    np.dot(
        weights.T,
        np.dot(
            cov_matrix,
            weights
        )
    )
)

st.write(
    f"Expected Portfolio Return: {portfolio_return:.2%}"
)

st.write(
    f"Expected Portfolio Risk: {portfolio_risk:.2%}"
)

# ====================================
# Monte-Carlo Simulation
# ====================================

st.subheader("Monte Carlo Portfolio Simulation")

risk_list = []
return_list = []
sharpe_list = []

for _ in range(5000):

    w = np.random.random(len(tickers))
    w /= np.sum(w)

    ret = np.sum(mean_returns * w)

    risk = np.sqrt(
        np.dot(
            w.T,
            np.dot(
                cov_matrix,
                w
            )
        )
    )

    sharpe = ret / risk

    risk_list.append(risk)
    return_list.append(ret)
    sharpe_list.append(sharpe)

simulation_df = pd.DataFrame({
    "Risk": risk_list,
    "Return": return_list,
    "Sharpe": sharpe_list
})

st.scatter_chart(
    simulation_df,
    x="Risk",
    y="Return"
)

# ====================================
# Bestes Portfolio
# ====================================

max_sharpe = simulation_df["Sharpe"].max()

best_portfolio = simulation_df[
    simulation_df["Sharpe"] == max_sharpe
]

st.subheader("Best Portfolio Found")

st.dataframe(best_portfolio)