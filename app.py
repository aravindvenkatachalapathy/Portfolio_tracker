import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Portfolio AI Tracker")

def compute_rsi(data, window=14):
    if len(data) < window + 1:
        return pd.Series([50.0]*len(data), index=data.index)
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

@st.cache_data(ttl=3600)
def load_portfolio(path):
    df = pd.read_excel(path, header=22)
    df = df.dropna(subset=['Symbol'])
    df = df[df['Symbol'] != 'Total']
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_history(ticker, period="5y"):
    # Try NSE first
    data = yf.download(ticker, period=period, interval="1d", progress=False)
    
    # If empty and it's an NSE ticker, smartly fallback to BSE
    if data.empty and ticker.endswith(".NS"):
        bse_ticker = ticker.replace(".NS", ".BO")
        data = yf.download(bse_ticker, period=period, interval="1d", progress=False)
        
    return data

def get_yahoo_ticker(symbol):
    # Strip Kite suffixes like -E (ETFs), -GB (Gold Bonds) safely
    # E.g., GOLDBEES-E -> GOLDBEES. Except keep MCDOWELL-N.
    if '-' in symbol and not symbol.endswith("-N"):
        symbol = symbol.split('-')[0]
    return f"{symbol}.NS"

def analyze_stock(history, symbol, avg_price=0.0):
    if history.empty or len(history) < 2:
        return None
        
    close_prices = history['Close'].squeeze()
    
    # Gracefully calculate even if there is shorter history
    sma_50 = close_prices.rolling(window=50, min_periods=1).mean()
    sma_200 = close_prices.rolling(window=200, min_periods=1).mean()
    rsi = compute_rsi(close_prices)
    
    current_price = float(close_prices.iloc[-1])
    current_rsi = float(rsi.iloc[-1])
    current_sma50 = float(sma_50.iloc[-1])
    current_sma200 = float(sma_200.iloc[-1])
    
    pnl_pct = ((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0
    
    signal = "⚪ HOLD"
    color = "gray"
    reason = "No strong trend indicators or limited historical data. Maintain position."
    
    # We only apply complex buy/sell indicators if the length is decent
    if len(close_prices) > 30:
        if current_price < current_sma50 and current_rsi < 35:
            signal = "🟢 BUY (Accumulate/Enter)"
            color = "green"
            reason = f"Significantly undervalued. RSI is oversold ({current_rsi:.1f}) and stock is heavily discounted."
        elif avg_price > 0 and pnl_pct > 15 and current_rsi > 70:
            signal = "🔴 SELL (Book Profit)"
            color = "red"
            reason = f"You are up {pnl_pct:.1f}%. The stock is heavily overbought (RSI {current_rsi:.1f}). Take profits now."
        elif current_rsi > 75:
            signal = "🔴 SELL (Overbought)"
            color = "red"
            reason = f"Stock is significantly overvalued with RSI at {current_rsi:.1f}. Not a safe time to buy."
        elif avg_price > 0 and pnl_pct < -20 and current_price < current_sma200 and len(close_prices) > 200:
            signal = "🔴 SELL (Cut Losses)"
            color = "red"
            reason = f"Stock is down {pnl_pct:.1f}% and trading below 200-day trendline. Exit to prevent deeper loss."
            
    # For very low history assets like new SGBs or recent listings, give default reason
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
        "sma_50_series": sma_50,
        "sma_200_series": sma_200
    }

def render_stock_card(analysis, avg_price=0.0):
    symbol = analysis['symbol']
    history = analysis['history']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=analysis['close_prices'], mode='lines', name='Price', line=dict(color='#2E86C1')))
    fig.add_trace(go.Scatter(x=history.index, y=analysis['sma_50_series'], mode='lines', line=dict(dash='dot', color='orange'), name='50-day SMA'))
    fig.add_trace(go.Scatter(x=history.index, y=analysis['sma_200_series'], mode='lines', line=dict(dash='dot', color='purple'), name='200-day SMA'))
    
    if avg_price > 0:
        fig.add_hline(y=avg_price, line_dash="dash", line_color="green" if analysis['pnl_pct'] > 0 else "red", annotation_text=f"Your Avg Price (₹{avg_price:.2f})")
        
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Date", yaxis_title="Price (INR)")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(f"🔹 {symbol}")
        st.metric("Current Price", f"₹{analysis['current_price']:.2f}")
        if avg_price > 0:
            st.markdown(f"**P&L:** <span style='font-size:24px; font-weight:bold; color:{'green' if analysis['pnl_pct']>0 else 'red'}'>{analysis['pnl_pct']:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"**RSI (14-day)**: {analysis['current_rsi']:.1f}")
        
        st.markdown(
            f"<div style='background-color: #2b2b2b; padding: 15px; border-radius: 10px; border-left: 5px solid {analysis['color']};'>"
            f"<b style='color:{analysis['color']}; font-size: 18px;'>{analysis['signal']}</b><br/>"
            f"<span style='color: white;'>{analysis['reason']}</span>"
            f"</div>", 
            unsafe_allow_html=True
        )
    with col2:
        st.plotly_chart(fig, use_container_width=True)

# Data Structures for Market Watchlists
CATEGORIES = {
    "Technology / IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS", "LTTS.NS"],
    "Banking & Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "PAYTM.NS"],
    "Auto / Manufacturing": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJAUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BOSCHLTD.NS", "CUMMINSIND.NS"],
    "Energy / Oil & Gas": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "IOC.NS", "BPCL.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "GAIL.NS"],
    "FMCG / Consumer": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "COLPAL.NS", "UBL.NS", "MCDOWELL-N.NS"]
}

# Flattens all categories to make the top 50 array
TOP_50_STOCKS = [stock for category in CATEGORIES.values() for stock in category]

st.title("Nivetha's Portfolio Tracker & Recommendation Engine")

excel_path = r"C:\Users\Aravind\control\Porfolio\holdings-UB7034.xlsx"
try:
    portfolio = load_portfolio(excel_path)
except Exception as e:
    st.error(f"Error reading Excel file: {e}. Make sure the portofolio file is loaded correctly.")
    st.stop()

# Layout Tabs
tab1, tab2, tab3 = st.tabs([" Nivetha's Portfolio", " Top 50 Market Watch", " Stock Categories"])

with tab1:
    st.header("Nivetha's Personal Holdings")
    #st.info("Analyzing your exported Kite portfolio to generate intelligent Buy/Sell/Hold signals based on your average buy price.")
    
    summary_container = st.empty()
    
    my_recommendations = []
    total_invested = 0.0
    total_current_value = 0.0
    
    for index, row in portfolio.iterrows():
        symbol = str(row['Symbol']).strip()
        avg_price = float(row.get('Average Price', 0))
        
        # Calculate exactly how many shares are owned
        qty_available = float(row.get('Quantity Available', 0)) if pd.notna(row.get('Quantity Available')) else 0
        qty_long_term = float(row.get('Quantity Long Term', 0)) if pd.notna(row.get('Quantity Long Term')) else 0
        total_qty = qty_available + qty_long_term
        
        # Use our safely stripped symbol for Yahoo Finance
        ticker = get_yahoo_ticker(symbol)
        
        st.write("---")
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                hist = fetch_history(ticker)
                analysis = analyze_stock(hist, symbol, avg_price)
                if analysis:
                    render_stock_card(analysis, avg_price)
                    my_recommendations.append({"Symbol": symbol, "Signal": analysis['signal'], "Reason": analysis['reason']})
                    
                    # Accumulate portfolio metrics
                    total_invested += (avg_price * total_qty)
                    total_current_value += (analysis['current_price'] * total_qty)
                    
                else:
                    st.warning(f"Could not compute analysis for {symbol}")
            except Exception as e:
                st.error(f"Failed to analyze {symbol}: {str(e)}")

    # Render top summary scoreboard now that we have calculated all the totals
    with summary_container.container():
        total_pnl = total_current_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Invested", f"₹{total_invested:,.2f}")
        col2.metric("💳 Current Value", f"₹{total_current_value:,.2f}")
        col3.metric("📈 Total Profit & Loss", f"₹{total_pnl:,.2f}", f"{total_pnl_pct:,.2f}%")
        st.write("---")

    if my_recommendations:
        st.markdown("### 📋 Executive Summary")
        st.table(pd.DataFrame(my_recommendations).set_index('Symbol'))

with tab2:
    st.header("Top 50 Watchlist Recommendations")
    st.write("These are the most prominent stocks in the Indian market. We are scanning them in real-time to find stocks currently flashing **BUY** signals.")
    
    if st.button("Scan Top 50 for Buy Opportunities"):
        progress_bar = st.progress(0)
        strong_buys = []
        
        for i, ticker in enumerate(TOP_50_STOCKS):
            hist = fetch_history(ticker)
            analysis = analyze_stock(hist, ticker, 0) # 0 avg price since we don't own it
            if analysis and "BUY" in analysis["signal"]:
                strong_buys.append(analysis)
            progress_bar.progress((i + 1) / len(TOP_50_STOCKS))
            
        progress_bar.empty()
        
        if strong_buys:
            st.success(f"Found {len(strong_buys)} strong BUY opportunities among the top 50!")
            for analysis in strong_buys:
                st.write("---")
                render_stock_card(analysis, 0.0)
        else:
            st.info("No strong buy signals found right now in the Top 50. The market might be running too hot.")

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
                render_stock_card(analysis, 0.0)
            else:
                st.warning(f"Failed to fetch data for {ticker}")
