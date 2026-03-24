import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import os
import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import json

os.environ["GOOGLE_API_KEY"] = "AIzaSyAWrsppAxu_7wukeHFhVvmg8C0GrHu0v1U"

st.set_page_config(layout="wide", page_title="Nivetha's Live Kite Tracker")

# ========= TECHNICAL INDICATORS =========

def compute_rsi(data, window=14):
    if len(data) < window + 1:
        return pd.Series([50.0] * len(data), index=data.index)
    diff = data.diff(1).dropna()
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)

    gain_series = pd.Series(gain, index=diff.index)
    loss_series = pd.Series(loss, index=diff.index)

    avg_gain = gain_series.rolling(window=window, min_periods=1).mean()
    avg_loss = loss_series.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close_prices, fast=12, slow=26, signal=9):
    if len(close_prices) < slow + signal:
        return (
            pd.Series([0.0] * len(close_prices), index=close_prices.index),
            pd.Series([0.0] * len(close_prices), index=close_prices.index),
            pd.Series([0.0] * len(close_prices), index=close_prices.index),
        )
    ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_bollinger_bands(close_prices, window=20, num_std=2):
    if len(close_prices) < window:
        sma = close_prices.rolling(window=window, min_periods=1).mean()
        upper = sma
        lower = sma
    else:
        sma = close_prices.rolling(window=window).mean()
        std = close_prices.rolling(window=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
    return sma, upper, lower


# ========= FUNDAMENTAL FILTER (Buffett-ish heuristic) =========

def buffett_score(info: dict) -> dict:
    pe = info.get("trailingPE")
    roe = info.get("returnOnEquity")
    profit_margin = info.get("profitMargins")
    debt_to_eq = info.get("debtToEquity")
    earnings_growth = info.get("earningsQuarterlyGrowth")

    score = 0
    reasons = []

    if pe is not None and pe > 0:
        if pe < 15:
            score += 2
            reasons.append("Attractive valuation (PE < 15).")
        elif pe < 25:
            score += 1
            reasons.append("Reasonable valuation (PE < 25).")
        else:
            reasons.append("Rich valuation (PE is high).")

    if roe is not None:
        if roe > 0.15:
            score += 2
            reasons.append("Strong return on equity (>15%).")
        elif roe > 0.10:
            score += 1
            reasons.append("Decent return on equity (>10%).")
        else:
            reasons.append("Weak ROE (<10%).")

    if profit_margin is not None:
        if profit_margin > 0.15:
            score += 2
            reasons.append("Healthy profit margins (>15%).")
        elif profit_margin > 0.08:
            score += 1
            reasons.append("Okay profit margins (>8%).")
        else:
            reasons.append("Thin profit margins.")

    if debt_to_eq is not None:
        if debt_to_eq < 50:
            score += 2
            reasons.append("Conservative leverage (Debt/Equity < 50).")
        elif debt_to_eq < 100:
            score += 1
            reasons.append("Moderate leverage.")
        else:
            reasons.append("High leverage (Debt/Equity > 100).")

    if earnings_growth is not None:
        if earnings_growth > 0.10:
            score += 1
            reasons.append("Earnings growing >10% YoY.")
        elif earnings_growth < 0:
            reasons.append("Earnings contracting YoY.")

    if score >= 6:
        label = "Buffett-grade (Strong Fundamentals)"
        color = "green"
    elif score >= 3:
        label = "Reasonable Fundamentals"
        color = "orange"
    else:
        label = "Weak Fundamentals"
        color = "red"

    return {
        "score": score,
        "label": label,
        "color": color,
        "reasons": reasons,
        "pe": pe,
        "roe": roe,
        "profit_margin": profit_margin,
        "debt_to_eq": debt_to_eq,
        "earnings_growth": earnings_growth,
    }


# ========= DATA LOADING (LIVE KITE API MCP INTEGRATION) =========

@st.cache_data(ttl=300) # Ping Zerodha every 5 mins automatically
def load_portfolio():
    async def fetch_holdings_from_mcp():
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "mcp-remote", "https://mcp.kite.trade/mcp"]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Execute the Zerodha kite API proxy directly requesting holdings!
                result = await session.call_tool("mcp_kite_get_holdings", {})
                return json.loads(result.content[0].text)

    try:
        raw_json = asyncio.run(fetch_holdings_from_mcp())
    except Exception as e:
        st.error(f"Failed to connect to Kite MCP Server: {e}")
        return pd.DataFrame()
        
    # Kite API returns `{"status": "success", "data": [{"tradingsymbol": "...", "average_price": ...}, ...]}`
    if isinstance(raw_json, dict) and "data" in raw_json:
        raw_holdings = raw_json["data"]
    elif isinstance(raw_json, list):
        raw_holdings = raw_json
    else:
        raw_holdings = []

    mapped_data = []
    for h in raw_holdings:
        mapped_data.append({
            "Symbol": str(h.get("tradingsymbol", "")),
            "Average Price": float(h.get("average_price", 0.0)),
            "Quantity Available": float(h.get("quantity", 0)),  
            "Quantity Long Term": 0.0,
        })
        
    df = pd.DataFrame(mapped_data)
    if not df.empty:
        df = df[df["Symbol"] != "Total"]
        df = df[df["Symbol"] != ""]
        df = df.dropna(subset=["Symbol"])
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_history(ticker, period="5y"):
    data = yf.download(ticker, period=period, interval="1d", progress=False)
    if data.empty and ticker.endswith(".NS"):
        bse_ticker = ticker.replace(".NS", ".BO")
        data = yf.download(bse_ticker, period=period, interval="1d", progress=False)
    return data

def get_yahoo_ticker(symbol):
    if "-" in symbol and not symbol.endswith("-N"):
        symbol = symbol.split("-")[0]
    return f"{symbol}.NS"


# ========= AI SUMMARY =========

def get_ai_summary(symbol, analysis, buffett_fundamentals):
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Google API key not configured."
    
    genai.configure(api_key=api_key)
    
    macd_now = analysis["macd_line"].iloc[-1]
    macd_signal_now = analysis["macd_signal"].iloc[-1]
    macd_hist_now = analysis["macd_hist"].iloc[-1]

    prompt = f"""
You are an Indian equity analyst who follows Warren Buffett-style fundamentals plus technicals.
Analyze this stock and give a concise view.

Symbol: {symbol}
Current price: {analysis['current_price']:.2f}
P&L (%): {analysis['pnl_pct']:.2f}
RSI(14): {analysis['current_rsi']:.2f}
MACD latest: {macd_now:.4f}
MACD signal: {macd_signal_now:.4f}
MACD histogram: {macd_hist_now:.4f}
50-day SMA: {analysis['sma_50']:.2f}
200-day SMA: {analysis['sma_200']:.2f}
Bollinger mid: {analysis['bb_mid']:.2f}
Bollinger upper: {analysis['bb_upper']:.2f}
Bollinger lower: {analysis['bb_lower']:.2f}
Existing rule-based signal: {analysis['signal']}
Existing reason: {analysis['reason']}

Buffett-style fundamental view:
Label: {buffett_fundamentals.get('label')}
Score: {buffett_fundamentals.get('score')}
PE: {buffett_fundamentals.get('pe')}
ROE: {buffett_fundamentals.get('roe')}
Profit margin: {buffett_fundamentals.get('profit_margin')}
Debt/Equity: {buffett_fundamentals.get('debt_to_eq')}
Earnings growth YoY: {buffett_fundamentals.get('earnings_growth')}
Key comments: {"; ".join(buffett_fundamentals.get('reasons', []))}

Write:
1) 2–3 sentence summary of technicals (trend, momentum, volatility) clearly mentioning MACD, RSI, and Bollinger Bands.
2) 2–3 sentence summary of fundamentals in plain language (quality, valuation, leverage).
3) A clear action suggestion for a long-term investor: Accumulate / Hold / Partial Profit Booking / Avoid / Exit, with 1 sentence why.
Keep total answer under 180 words, neutral tone, no hype.
"""

    model = genai.GenerativeModel("gemini-2.5-flash")  
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error while calling Gemini: {e}"


# ========= CORE ANALYSIS =========

def analyze_stock(history, symbol, avg_price=0.0):
    if history.empty or len(history) < 2:
        return None

    close_prices = history["Close"].squeeze()

    # --- NEW RISK METRICS ---
    # Calculate daily percentage changes
    daily_returns = close_prices.pct_change().dropna()
    
    # 1. Percentage of Down Days (Looking at the last 2 years / ~504 trading days)
    recent_returns = daily_returns.tail(504) if len(daily_returns) > 504 else daily_returns
    pct_down_days = (recent_returns < 0).sum() / len(recent_returns) * 100 if not recent_returns.empty else 0.0

    # 2. Value at Risk (VaR) at 95% Confidence (Historical Method)
    # This finds the 5th percentile of worst days. 
    var_95 = np.percentile(recent_returns, 5) * 100 if not recent_returns.empty else 0.0
    # ------------------------

    sma_50_series = close_prices.rolling(window=50, min_periods=1).mean()
    sma_200_series = close_prices.rolling(window=200, min_periods=1).mean()
    rsi_series = compute_rsi(close_prices)
    macd_line, macd_signal, macd_hist = compute_macd(close_prices)
    bb_mid_series, bb_upper_series, bb_lower_series = compute_bollinger_bands(close_prices)

    current_price = float(close_prices.iloc[-1])
    current_rsi = float(rsi_series.iloc[-1])
    current_sma50 = float(sma_50_series.iloc[-1])
    current_sma200 = float(sma_200_series.iloc[-1])
    current_bb_mid = float(bb_mid_series.iloc[-1])
    current_bb_upper = float(bb_upper_series.iloc[-1])
    current_bb_lower = float(bb_lower_series.iloc[-1])

    pnl_pct = ((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0

    signal = "⚪ HOLD"
    color = "gray"
    reason = "No strong trend indicators or limited historical data. Maintain position."

    if len(close_prices) > 30:
        if current_price < current_sma50 and current_rsi < 35:
            signal = "🟢 BUY (Accumulate/Enter)"
            color = "green"
            reason = (f"Significantly undervalued. RSI is oversold ({current_rsi:.1f}) and "
                      f"price is below 50-DMA.")
        elif avg_price > 0 and pnl_pct > 15 and current_rsi > 70:
            signal = "🔴 SELL (Book Profit)"
            color = "red"
            reason = (f"You are up {pnl_pct:.1f}%. RSI is overbought ({current_rsi:.1f}); "
                      f"consider booking partial profits.")
        elif current_rsi > 75:
            signal = "🔴 SELL (Overbought)"
            color = "red"
            reason = (f"Stock is significantly overbought with RSI at {current_rsi:.1f}. "
                      f"Risk of pullback is high.")
        elif avg_price > 0 and pnl_pct < -20 and current_price < current_sma200 and len(close_prices) > 200:
            signal = "🔴 SELL (Cut Losses)"
            color = "red"
            reason = (f"Stock is down {pnl_pct:.1f}% and trades below 200-DMA; "
                      f"trend is weak, consider exiting.")

    if len(close_prices) < 30:
        reason = "Newly listed asset, recent bond issue, or ETF with scarce data. Monitoring value."

    return {
        "symbol": symbol,
        "current_price": current_price,
        "current_rsi": current_rsi,
        "sma_50": current_sma50,
        "sma_200": current_sma200,
        "pnl_pct": pnl_pct,
        "signal": signal,
        "color": color,
        "reason": reason,
        "history": history,
        "close_prices": close_prices,
        "sma_50_series": sma_50_series,
        "sma_200_series": sma_200_series,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_mid_series": bb_mid_series,
        "bb_upper_series": bb_upper_series,
        "bb_lower_series": bb_lower_series,
        "bb_mid": current_bb_mid,
        "bb_upper": current_bb_upper,
        "bb_lower": current_bb_lower,
        # Pass the new metrics to the frontend
        "pct_down_days": pct_down_days,
        "var_95": var_95,
    }


# ========= RENDERING =========
def render_stock_card(analysis, avg_price=0.0, fundamentals=None, ai_enabled=True):
    symbol = analysis["symbol"]
    history = analysis["history"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
        subplot_titles=("Price, MAs & Bollinger Bands", "MACD"),
    )

    fig.add_trace(go.Scatter(x=history.index, y=analysis["close_prices"], mode="lines", name="Price", line=dict(color="#2E86C1")), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=analysis["sma_50_series"], mode="lines", name="50-day SMA", line=dict(dash="dot", color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=analysis["sma_200_series"], mode="lines", name="200-day SMA", line=dict(dash="dot", color="purple")), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=analysis["bb_upper_series"], mode="lines", name="Upper BB", line=dict(color="gray", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=analysis["bb_mid_series"], mode="lines", name="Middle BB", line=dict(color="lightgray", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=analysis["bb_lower_series"], mode="lines", name="Lower BB", line=dict(color="gray", width=1)), row=1, col=1)

    if avg_price > 0:
        fig.add_hline(y=avg_price, line_dash="dash", line_color="green" if analysis["pnl_pct"] > 0 else "red", annotation_text=f"Your Avg Price (₹{avg_price:.2f})", row=1, col=1)

    fig.add_trace(go.Scatter(x=history.index, y=analysis["macd_line"], mode="lines", name="MACD", line=dict(color="cyan")), row=2, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=analysis["macd_signal"], mode="lines", name="Signal", line=dict(color="magenta")), row=2, col=1)
    fig.add_trace(go.Bar(x=history.index, y=analysis["macd_hist"], name="Histogram", marker_color=np.where(analysis["macd_hist"] >= 0, "green", "red")), row=2, col=1)

    fig.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Date", yaxis_title="Price (INR)")
    fig.update_yaxes(title_text="MACD", row=2, col=1)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(f"🔹 {symbol}")
        st.metric("Current Price", f"₹{analysis['current_price']:.2f}")
        if avg_price > 0:
            st.markdown(f"**P&L:** <span style='font-size:24px; font-weight:bold; color:{'green' if analysis['pnl_pct']>0 else 'red'}'>{analysis['pnl_pct']:.2f}%</span>", unsafe_allow_html=True)
        
        st.markdown(f"**RSI (14-day)**: {analysis['current_rsi']:.1f}")
        st.markdown(f"**Bollinger Position**: Price vs Bands<br>- Upper: ₹{analysis['bb_upper']:.2f}<br>- Middle: ₹{analysis['bb_mid']:.2f}<br>- Lower: ₹{analysis['bb_lower']:.2f}", unsafe_allow_html=True)

        # --- NEW RISK DISPLAY SECTION ---
        st.markdown("### Risk Analysis (Last 2 Yrs)")
        st.markdown(f"📉 **Down Days:** {analysis['pct_down_days']:.1f}%")
        st.markdown(f"⚠️ **95% Daily VaR:** {analysis['var_95']:.2f}% *(Max expected daily loss 95% of the time)*")
        # --------------------------------

        st.markdown(f"<div style='background-color: #2b2b2b; padding: 15px; border-radius: 10px; border-left: 5px solid {analysis['color']}; margin-top: 15px;'><b style='color:{analysis['color']}; font-size: 18px;'>{analysis['signal']}</b><br/><span style='color: white;'>{analysis['reason']}</span></div>", unsafe_allow_html=True)

        if fundamentals:
            st.markdown("### Buffett-style Fundamentals")
            st.markdown(f"<div style='background-color:#1c1c1c;padding:10px;border-radius:8px;border-left:4px solid {fundamentals['color']};'><b style='color:{fundamentals['color']};'>{fundamentals['label']}</b><br>Score: {fundamentals['score']}<br>" + "<br>".join(fundamentals["reasons"]) + "</div>", unsafe_allow_html=True)

    with col2:
        st.plotly_chart(fig, use_container_width=True)

    if ai_enabled:
        with st.expander("🤖 AI Summary (MACD + RSI + Bollinger + Buffett View)"):
            if st.button("Generate AI View", key=f"ai_{symbol}"):
                with st.spinner("Generating AI summary..."):
                    text = get_ai_summary(symbol, analysis, fundamentals or {})
                st.markdown(text)


# ========= WATCHLIST STRUCTURES =========

CATEGORIES = {
    "Technology / IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS", "LTTS.NS"],
    "Banking & Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "PAYTM.NS"],
    "Auto / Manufacturing": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJAUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BOSCHLTD.NS", "CUMMINSIND.NS"],
    "Energy / Oil & Gas": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "IOC.NS", "BPCL.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "GAIL.NS"],
    "FMCG / Consumer": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "COLPAL.NS", "UBL.NS", "MCDOWELL-N.NS"],
}

TOP_50_STOCKS = [stock for category in CATEGORIES.values() for stock in category]


# ========= APP LAYOUT =========

st.title("Nivetha's Live Kite Dashboard 🚀")

try:
    portfolio = load_portfolio()
except Exception as e:
    st.error(f"Error fetching portfolio from Kite MCP Server: {e}.")
    st.stop()

if portfolio.empty:
    st.warning("Your Zerodha Kite portfolio currently has no holdings, or the server failed to fetch them.")

tab1, tab2, tab3 = st.tabs([" Nivetha's Portfolio", " Top 50 Market Watch", " Stock Categories"])

# ----- TAB 1: Portfolio -----
with tab1:
    st.header("Nivetha's Personal Holdings (Live Data)")
    summary_container = st.empty()
    my_recommendations = []
    total_invested = 0.0
    total_current_value = 0.0
    rows_for_table = []
    analysis_by_symbol = {}
    fundamentals_by_symbol = {}

    for index, row in portfolio.iterrows():
        symbol = str(row["Symbol"]).strip()
        avg_price = float(row.get("Average Price", 0))
        qty_available = float(row.get("Quantity Available", 0)) if pd.notna(row.get("Quantity Available")) else 0
        qty_long_term = float(row.get("Quantity Long Term", 0)) if pd.notna(row.get("Quantity Long Term")) else 0
        total_qty = qty_available + qty_long_term

        if total_qty <= 0:
            continue

        ticker_str = get_yahoo_ticker(symbol)

        with st.spinner(f"Analyzing {symbol}..."):
            try:
                hist = fetch_history(ticker_str)
                analysis = analyze_stock(hist, symbol, avg_price)
                if analysis:
                    ticker_yf = yf.Ticker(ticker_str)
                    info = getattr(ticker_yf, "info", {})
                    buffett = buffett_score(info or {})

                    analysis_by_symbol[symbol] = (analysis, avg_price)
                    fundamentals_by_symbol[symbol] = buffett

                    invested_amt = avg_price * total_qty
                    current_value = analysis["current_price"] * total_qty
                    pnl_value = current_value - invested_amt
                    pnl_pct = (pnl_value / invested_amt * 100) if invested_amt > 0 else 0

                    rows_for_table.append({
                        "Symbol": symbol, "Quantity": total_qty, "Avg Price (₹)": avg_price,
                        "Invested (₹)": invested_amt, "Current Price (₹)": analysis["current_price"],
                        "Current Value (₹)": current_value, "P&L (₹)": pnl_value,
                        "P&L (%)": pnl_pct, "Decision": analysis["signal"], "Buffett View": buffett["label"],
                    })

                    my_recommendations.append({"Symbol": symbol, "Signal": analysis["signal"], "Reason": analysis["reason"]})
                    total_invested += invested_amt
                    total_current_value += current_value

            except Exception as e:
                st.error(f"Failed to analyze {symbol}: {str(e)}")

    with summary_container.container():
        total_pnl = total_current_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Invested", f"₹{total_invested:,.2f}")
        col2.metric("💳 Current Value", f"₹{total_current_value:,.2f}")
        col3.metric("📈 Total Profit & Loss", f"₹{total_pnl:,.2f}", f"{total_pnl_pct:,.2f}%")
        st.write("---")

    if rows_for_table:
        st.subheader("Portfolio Overview (Table)")
        df_view = pd.DataFrame(rows_for_table).set_index("Symbol")
        st.dataframe(df_view.style.format({
            "Quantity": "{:,.0f}", "Avg Price (₹)": "₹{:,.2f}", "Invested (₹)": "₹{:,.2f}",
            "Current Price (₹)": "₹{:,.2f}", "Current Value (₹)": "₹{:,.2f}",
            "P&L (₹)": "₹{:,.2f}", "P&L (%)": "{:,.2f}%",
        }), use_container_width=True, height=420)

        st.markdown("### Details for Selected Stock")
        selected_symbol = st.selectbox("Choose a stock from your portfolio to view chart, fundamentals & AI summary", options=list(analysis_by_symbol.keys()))

        if selected_symbol:
            st.write("---")
            analysis, avg_price = analysis_by_symbol[selected_symbol]
            buffett = fundamentals_by_symbol.get(selected_symbol)
            render_stock_card(analysis, avg_price, fundamentals=buffett, ai_enabled=True)

    if my_recommendations:
        st.markdown("### 📋 Executive Summary")
        st.table(pd.DataFrame(my_recommendations).set_index("Symbol"))

# ----- TAB 2: Top 50 Watch -----
with tab2:
    st.header("Top 50 Watchlist Recommendations")
    st.write("These are the most prominent stocks in the Indian market. We are scanning them in real-time to find stocks currently flashing **BUY** signals.")

    if st.button("Scan Top 50 for Buy Opportunities"):
        progress_bar = st.progress(0)
        strong_buys = []

        for i, ticker in enumerate(TOP_50_STOCKS):
            hist = fetch_history(ticker)
            analysis = analyze_stock(hist, ticker, 0)
            if analysis and "BUY" in analysis["signal"]:
                # Fetch fundamentals so AI summary works properly
                ticker_yf = yf.Ticker(ticker)
                info = getattr(ticker_yf, "info", {})
                buffett = buffett_score(info or {})
                strong_buys.append((analysis, buffett))
            progress_bar.progress((i + 1) / len(TOP_50_STOCKS))

        progress_bar.empty()

        if strong_buys:
            st.success(f"Found {len(strong_buys)} strong BUY opportunities among the top 50!")
            for analysis, buffett in strong_buys:
                st.write("---")
                # AI enabled and fundamentals passed
                render_stock_card(analysis, 0.0, fundamentals=buffett, ai_enabled=True) 
        else:
            st.info("No strong buy signals found right now in the Top 50. The market might be running too hot.")

# ----- TAB 3: Sector Categories -----
with tab3:
    st.header("Sector-Wise Top 10 Tracking")
    st.write("View the 10 most influential stocks segmented by sector. Understand where the momentum is flowing.")

    selected_category = st.selectbox("Select a Sector to Analyze", list(CATEGORIES.keys()))
    st.write(f"### Analyzing 10 Leaders in {selected_category}")

    for ticker in CATEGORIES[selected_category]:
        st.write("---")
        with st.spinner(f"Fetching data for {ticker}..."):
            hist = fetch_history(ticker)
            analysis = analyze_stock(hist, ticker, 0)
            if analysis:
                # Fetch fundamentals so AI summary works properly
                ticker_yf = yf.Ticker(ticker)
                info = getattr(ticker_yf, "info", {})
                buffett = buffett_score(info or {})
                # AI enabled and fundamentals passed
                render_stock_card(analysis, 0.0, fundamentals=buffett, ai_enabled=True) 
            else:
                st.warning(f"Failed to fetch data for {ticker}")
