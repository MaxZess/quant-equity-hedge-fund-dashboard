import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Max Zessinger Quant Lab",
    page_icon="📈",
    layout="wide"
)

# ==================================================
# DESIGN
# ==================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background-color:#0E1117;
}

h1,h2,h3{
    color:#00D4FF;
}

[data-testid="metric-container"]{
    background-color:#1B2430;
    border:1px solid #2D3748;
    border-radius:12px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.title("🚀 Max Zessinger Quant Lab")

st.caption(
    "Portfolio Optimization • Risk Analytics • Quantitative Finance"
)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Dashboard Settings")

start_year = st.sidebar.slider(
    "Start Year",
    2018,
    2025,
    2020
)

tickers = st.sidebar.multiselect(
    "Select Stocks",
    [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "JPM",
        "V",
        "NFLX"
    ],
    default=[
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL"
    ]
)

st.sidebar.success(
    "Created by Maximilian Zessinger"
)

# ==================================================
# CHECK
# ==================================================

if len(tickers) < 2:
    st.warning("Please select at least 2 stocks.")
    st.stop()

# ==================================================
# DATA
# ==================================================

data = yf.download(
    tickers,
    start=f"{start_year}-01-01",
    auto_adjust=True
)["Close"]

returns = data.pct_change().dropna()

# ==================================================
# BASIC METRICS
# ==================================================

portfolio_returns = returns.mean(axis=1)

annual_return = portfolio_returns.mean() * 252

annual_volatility = (
    portfolio_returns.std()
    * np.sqrt(252)
)

sharpe_ratio = annual_return / annual_volatility

quant_score = max(
    0,
    min(sharpe_ratio * 50, 100)
)

# ==================================================
# TOP KPIs
# ==================================================

col1, col2, col3, col4 = st.columns(4)

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

col4.metric(
    "Quant Score",
    f"{quant_score:.1f}/100"
)

# ==================================================
# CHARTS
# ==================================================

st.subheader("📈 Stock Prices")
st.line_chart(data)

st.subheader("📊 Daily Returns")
st.line_chart(returns)

# ==================================================
# PORTFOLIO
# ==================================================

num_assets = len(tickers)

weights = np.repeat(
    1 / num_assets,
    num_assets
)

mean_returns_series = returns.mean() * 252

mean_returns = mean_returns_series.values

cov_matrix = (
    returns.cov().values
    * 252
)

portfolio_return = np.sum(
    weights * mean_returns
)

portfolio_risk = np.sqrt(
    weights.T
    @ cov_matrix
    @ weights
)

st.subheader("💼 Portfolio Statistics")

c1, c2 = st.columns(2)

c1.metric(
    "Expected Portfolio Return",
    f"{portfolio_return:.2%}"
)

c2.metric(
    "Expected Portfolio Risk",
    f"{portfolio_risk:.2%}"
)

# ==================================================
# RISK
# ==================================================

var95 = np.percentile(
    portfolio_returns,
    5
)

cumulative = (
    1 + portfolio_returns
).cumprod()

running_max = cumulative.cummax()

drawdown = (
    cumulative - running_max
) / running_max

max_drawdown = drawdown.min()

r1, r2 = st.columns(2)

r1.metric(
    "Value at Risk (95%)",
    f"{var95:.2%}"
)

r2.metric(
    "Maximum Drawdown",
    f"{max_drawdown:.2%}"
)

# ==================================================
# MONTE CARLO
# ==================================================

st.subheader("🎯 Monte Carlo Portfolio Simulation")

risk_list = []
return_list = []
sharpe_list = []

for _ in range(5000):

    w = np.random.random(num_assets)
    w = w / np.sum(w)

    ret = np.sum(w * mean_returns)

    risk = np.sqrt(
        w.T
        @ cov_matrix
        @ w
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

fig = px.scatter(
    simulation_df,
    x="Risk",
    y="Return",
    color="Sharpe",
    color_continuous_scale="Viridis",
    title="Efficient Frontier Simulation"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================================
# BEST PORTFOLIO
# ==================================================

best_index = simulation_df["Sharpe"].idxmax()

best_portfolio = simulation_df.loc[
    best_index
]

st.subheader("🏆 Best Portfolio")

st.dataframe(
    pd.DataFrame(best_portfolio).T,
    width="stretch"
)

# ==================================================
# TOP STOCK
# ==================================================

best_stock = mean_returns_series.idxmax()

st.success(
    f"Top Performing Stock: {best_stock}"
)

# ==================================================
# ALLOCATION
# ==================================================

portfolio_df = pd.DataFrame({
    "Ticker": tickers,
    "Weight (%)": np.round(
        weights * 100,
        2
    )
})

st.subheader("📋 Portfolio Allocation")

st.dataframe(
    portfolio_df,
    width="stretch"
)

# ==================================================
# CORRELATION
# ==================================================

st.subheader("🔥 Correlation Matrix")

corr = returns.corr()

corr_fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(
    corr_fig,
    width="stretch"
)

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown("### 🔬 Research Summary")

st.write(
    "Professional quantitative equity research dashboard featuring "
    "portfolio analytics, Monte Carlo simulation, Sharpe Ratio analysis, "
    "Value at Risk, Maximum Drawdown and correlation analysis."
)