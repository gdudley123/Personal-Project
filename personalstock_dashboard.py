# ================================
# 📦 IMPORT LIBRARIES & SETUP
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Stock Analyzer Dashboard", layout="wide")
st.title("📊 Stock Analyzer Dashboard")

# ===================================
# ⏬ FUNCTION TO FETCH STOCK DATA
# ===================================
@st.cache_data
def fetch_stock_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    return df.dropna()

# ========================================
# 🔍 FUNCTION TO DETECT CHART PATTERNS
# ========================================
def detect_candlestick_patterns(df):
    patterns = []
    for i in range(1, len(df)):
        o1, h1, l1, c1 = df.iloc[i - 1][["Open", "High", "Low", "Close"]]
        o2, h2, l2, c2 = df.iloc[i][["Open", "High", "Low", "Close"]]
        if c1 < o1 and c2 > o2 and o2 < c1 and c2 > o1:
            patterns.append(("Bullish Engulfing", df.index[i]))
        elif c1 > o1 and c2 < o2 and o2 > c1 and c2 < o1:
            patterns.append(("Bearish Engulfing", df.index[i]))
        elif abs(c2 - o2) < 0.1 * (h2 - l2):
            patterns.append(("Doji", df.index[i]))
        elif (c2 > o2) and ((l2 - min(c2, o2)) > 2 * abs(c2 - o2)) and ((h2 - max(c2, o2)) < abs(c2 - o2)):
            patterns.append(("Hammer", df.index[i]))
        elif (o2 > c2) and ((h2 - max(c2, o2)) > 2 * abs(c2 - o2)) and ((min(c2, o2) - l2) < abs(c2 - o2)):
            patterns.append(("Shooting Star", df.index[i]))
    return patterns

# ========================================
# 🧾 SIDEBAR INPUTS & STOCK SELECTION
# ========================================
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
range_options = {
    "Max": "max", "1 Year": "1y", "6 Months": "6mo", "3 Months": "3mo", "1 Month": "1mo"
}
selected_range = st.sidebar.selectbox("Select Date Range", list(range_options.keys()))
download_data = st.sidebar.button("Download Stock Data")

df = None
if download_data and ticker:
    df = fetch_stock_data(ticker, range_options[selected_range])
    st.sidebar.success(f"Loaded {selected_range} of data for {ticker}.")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Technical Charts", "Fundamental Analysis", "Pattern Recognition", "Portfolio Tracker"])

# ======================================
# 📈 TECHNICAL CHARTS & SIGNAL SECTION
# ======================================
if page == "Technical Charts":
    if df is None:
        st.warning("Please download stock data first.")
    else:
        # Calculate indicators
        df["SMA"] = df["Close"].rolling(window=20).mean()
        df["EMA"] = df["Close"].ewm(span=10, adjust=False).mean()
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["RSI Signal"] = np.where(df["RSI"] < 30, "🟢 BUY",
                                    np.where(df["RSI"] > 70, "🔴 SELL", "⚪ HOLD"))

        # Buy/Hold/Sell Recommendation
        latest_row = df.iloc[-1]
        rsi = float(latest_row["RSI"])
        price = float(latest_row["Close"])
        ema = float(latest_row["EMA"])
        sma = float(latest_row["SMA"])
        if rsi < 35 and ema > sma and price > ema:
            recommendation = "🟢 Strong Buy"
        elif rsi > 65 and ema < sma and price < ema:
            recommendation = "🔴 Strong Sell"
        else:
            recommendation = "⚪ Hold"
        st.subheader(f"📌 Recommendation: **{recommendation}**")

        # Price Chart with Signals
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                           low=df["Low"], close=df["Close"], name="Candlestick"))
        fig_price.add_trace(go.Scatter(x=df.index, y=df["SMA"], line=dict(color='blue'), name="SMA 20"))
        fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA"], line=dict(color='orange'), name="EMA 10"))
        fig_price.add_trace(go.Scatter(
            x=df.index[(df["RSI"] < 30) & (df["EMA"] > df["SMA"])],
            y=df["Close"][(df["RSI"] < 30) & (df["EMA"] > df["SMA"])],
            mode="markers", marker=dict(color="green", size=10, symbol="arrow-up"), name="Buy Signal"))
        fig_price.add_trace(go.Scatter(
            x=df.index[(df["RSI"] > 70) & (df["EMA"] < df["SMA"])],
            y=df["Close"][(df["RSI"] > 70) & (df["EMA"] < df["SMA"])],
            mode="markers", marker=dict(color="red", size=10, symbol="arrow-down"), name="Sell Signal"))
        fig_price.update_layout(title=f"{ticker} Price Chart", yaxis_title="Price", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_price, use_container_width=True)

        # RSI Chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color='purple'), name="RSI (14)"))
        fig_rsi.update_layout(title="Relative Strength Index (RSI)", yaxis_title="RSI",
                              xaxis_rangeslider_visible=False, yaxis=dict(range=[0, 100]),
                              shapes=[
                                  dict(type="line", xref="paper", x0=0, x1=1, y0=70, y1=70, line=dict(color="red", dash="dash")),
                                  dict(type="line", xref="paper", x0=0, x1=1, y0=30, y1=30, line=dict(color="green", dash="dash"))
                              ])
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.subheader("📊 RSI Signals (Last 10)")
        st.dataframe(df[["RSI", "RSI Signal"]].dropna().tail(10))

# ===================================
# 🧮 FUNDAMENTAL ANALYSIS SECTION
# ===================================
elif page == "Fundamental Analysis":
    if df is None:
        st.warning("Please download stock data first.")
    else:
        st.header(f"📊 Fundamental Analysis: {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.info
        st.subheader("Company Overview")
        for key in ["longName", "sector", "industry", "fullTimeEmployees", "website"]:
            st.write(f"**{key.title()}**: {info.get(key)}")

        st.subheader("Key Financial Metrics")
        metrics = {
            "Market Cap": info.get("marketCap"),
            "Revenue": info.get("totalRevenue"),
            "Net Income": info.get("netIncomeToCommon"),
            "EPS": info.get("trailingEps"),
            "Profit Margin": info.get("profitMargins"),
            "ROE": info.get("returnOnEquity"),
            "Debt/Equity": info.get("debtToEquity"),
            "P/E": info.get("trailingPE"),
            "PEG": info.get("pegRatio"),
            "EV/EBITDA": info.get("enterpriseToEbitda"),
            "P/B": info.get("priceToBook")
        }
        for k, v in metrics.items():
            st.write(f"**{k}**: {v}")

# =========================================
# 📐 PATTERN RECOGNITION (CHART PATTERNS)
# =========================================
elif page == "Pattern Recognition":
    if df is None:
        st.warning("Please download stock data first.")
    else:
        st.header(f"📐 Candlestick Pattern Recognition for {ticker}")
        patterns = detect_candlestick_patterns(df)
        if patterns:
            st.write(f"Found {len(patterns)} candlestick pattern(s):")
            for pattern, date in patterns[-20:][::-1]:
                st.markdown(f"- **{pattern}** on `{date.date()}`")
        else:
            st.info("No major candlestick patterns detected.")

# ====================================
# 📂 PORTFOLIO TRACKER + CSV EXPORT
# ====================================
elif page == "Portfolio Tracker":
    st.header("📂 Portfolio Tracker")
    try:
        portfolio = pd.read_csv("portfolio.csv")
    except:
        portfolio = pd.DataFrame(columns=["Ticker", "Type", "Price", "Shares", "Date"])

    st.subheader("Current Portfolio")
    st.dataframe(portfolio)

    if not portfolio.empty:
        csv = portfolio.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Portfolio as CSV", data=csv, file_name="my_portfolio.csv", mime="text/csv")

    st.subheader("Add Trade")
    with st.form("trade_form"):
        ticker_input = st.text_input("Ticker").upper()
        trade_type = st.selectbox("Type", ["Buy", "Sell"])
        price = st.number_input("Price", value=100.0)
        shares = st.number_input("Shares", value=10)
        date = st.date_input("Date", value=datetime.today())
        submitted = st.form_submit_button("Submit")
        if submitted:
            new_trade = pd.DataFrame([[ticker_input, trade_type, price, shares, date]], columns=portfolio.columns)
            portfolio = pd.concat([portfolio, new_trade], ignore_index=True)
            portfolio.to_csv("portfolio.csv", index=False)
            st.success("Trade added!")

    if not portfolio.empty:
        portfolio["Value"] = portfolio["Price"] * portfolio["Shares"]
        net = portfolio[portfolio["Type"] == "Buy"]["Value"].sum() - portfolio[portfolio["Type"] == "Sell"]["Value"].sum()
        st.metric("Net Investment", f"${net:,.2f}")
        st.metric("Total Value", f"${portfolio['Value'].sum():,.2f}")
        st.metric("PnL", f"${portfolio['Value'].sum() - net:,.2f}")
