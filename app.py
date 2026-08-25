import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Max Zessinger Quant Lab",
    page_icon="📈",
    layout="wide"
)

# ======================================
# CUSTOM CSS
# ======================================

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
    padding:15px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ======================================
# HEADER
# ======================================

st.title("🚀 Max Zessinger Quant Lab")

st.caption(
    "Portfolio Optimization | Risk Analytics | Quantitative Finance"
)

# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("Dashboard Settings")

start_year = st.sidebar.slider(
    "Analyse ab Jahr",
    2018,
    2025,
    2020
)

st.sidebar.success(
    "Created by Maximilian Zessinger"
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
        "JPM",
        "TSLA",
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

# ======================================
# DATA
# ======================================

data = yf.download(
    tickers,
    start=f"{start_year}-01-01",
    auto_adjust=True
)["Close"]

returns = data.pct_change().dropna()

portfolio_returns = returns.mean(axis=1)

annual_return = portfolio_returns.mean() * 252

annual_volatility = (
    portfolio_returns.std()
    * np.sqrt(252)
)

sharpe_ratio = annual_return / annual_volatility

# ======================================
# TOP KPIs
# ======================================

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

quant_score = sharpe_ratio * 50

col4.metric(
    "Quant Score",
    f"{quant_score:.1f}/100"
)

# ======================================
# STOCK PRICE CHART
# ======================================

st.subheader("📈 Stock Prices")

st.line_chart(data)

# ======================================
# DAILY RETURNS
# ======================================

st.subheader("📊 Daily Returns")

st.line_chart(returns)

# ======================================
# PORTFOLIO STATISTICS
# ======================================

weights = np.ones(len(tickers))
weights = weights / len(tickers)
portfolio_df = pd.DataFrame({
    "Ticker": tickers,
    "Weight (%)": weights * 100
})

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

st.subheader("💼 Portfolio Statistics")

stat1, stat2 = st.columns(2)

stat1.metric(
    "Expected Portfolio Return",
    f"{portfolio_return:.2%}"
)

stat2.metric(
    "Expected Portfolio Risk",
    f"{portfolio_risk:.2%}"
)

# ======================================
# VAR
# ======================================

var_95 = np.percentile(
    portfolio_returns,
    5
)

# ======================================
# MAX DRAWDOWN
# ======================================

cumulative = (
    1 + portfolio_returns
).cumprod()

running_max = cumulative.cummax()

drawdown = (
    cumulative - running_max
) / running_max

max_drawdown = drawdown.min()

risk1, risk2 = st.columns(2)

risk1.metric(
    "Value at Risk (95%)",
    f"{var_95:.2%}"
)

risk2.metric(
    "Maximum Drawdown",
    f"{max_drawdown:.2%}"
)

# ======================================
# MONTE CARLO
# ======================================

st.subheader("🎯 Monte Carlo Portfolio Simulation")

risk_list = []
return_list = []
sharpe_list = []

for _ in range(5000):

    w = np.random.random(
        len(tickers)
    )

    w /= np.sum(w)

    ret = np.sum(
        mean_returns * w
    )

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

# ======================================
# BEST PORTFOLIO
# ======================================

best_index = simulation_df[
    "Sharpe"
].idxmax()

best_portfolio = simulation_df.loc[
    best_index
]

st.subheader("🏆 Best Portfolio Found")

st.dataframe(
    pd.DataFrame(best_portfolio).T,
    use_container_width=True
)

# ======================================
# TOP STOCK
# ======================================

best_stock = mean_returns.idxmax()

st.success(
    f"Top Performing Stock: {best_stock}"
)

# ======================================
# ALLOCATION
# ======================================

portfolio_df = pd.DataFrame({
    "Ticker": tickers,
    "Weight (%)": weights * 100
})

st.subheader("📋 Portfolio Allocation")

st.dataframe(
    portfolio_df,
    use_container_width=True
)

# ======================================
# FOOTER
# ======================================

st.markdown("---")

st.markdown(
    "### 🔬 Research Summary"
)

st.write(
    "This dashboard analyzes equity performance, portfolio risk, "
    "Monte Carlo simulations, Value at Risk (VaR), Maximum Drawdown "
    "and portfolio allocation across major US technology stocks."
)
import plotly.express as px

st.subheader("🔥 Correlation Matrix")

corr = returns.corr()

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
