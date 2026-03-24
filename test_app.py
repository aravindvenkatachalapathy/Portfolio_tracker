import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import openai
import os
from datetime import datetime, date

# ── secrets ──────────────────────────────────────────────────────────────────
os.environ.setdefault("GOOGLE_API_KEY", "AIzaSyAWrsppAxu_7wukeHFhVvmg8C0GrHu0v1U")

st.set_page_config(layout="wide", page_title="Nivetha's Portfolio Tracker", page_icon="")


# ════════════════════════════════════════════════════════════════════════════
# 1.  AI FALLBACK ENGINE
# ════════════════════════════════════════════════════════════════════════════

def call_ai(prompt: str) -> str:
    gkey = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    okey = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    err = None
    if gkey:
        try:
            genai.configure(api_key=gkey)
            return genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt).text
        except Exception as e:
            err = e
    if okey:
        try:
            client = openai.OpenAI(api_key=okey)
            r = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content
        except Exception as e:
            err = e
    raise RuntimeError(f"All AI providers failed. Last error: {err}")


# ════════════════════════════════════════════════════════════════════════════
# 2.  TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    if len(series) < window + 1:
        return pd.Series([50.0] * len(series), index=series.index)
    diff = series.diff(1).dropna()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_g = gain.rolling(window, min_periods=1).mean()
    avg_l = loss.rolling(window, min_periods=1).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, sig=9):
    if len(series) < slow + sig:
        z = pd.Series([0.0] * len(series), index=series.index)
        return z, z, z
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal, macd - signal


def compute_macd_monthly(series: pd.Series, fast=12, slow=26, sig=9):
    monthly = series.resample("ME").last().dropna()
    if len(monthly) < slow + sig:
        return False, False
    ema_f  = monthly.ewm(span=fast, adjust=False).mean()
    ema_s  = monthly.ewm(span=slow, adjust=False).mean()
    macd   = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    bullish       = bool(macd.iloc[-1] > signal.iloc[-1])
    just_crossed  = bool(macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2])
    return bullish, just_crossed


def compute_bollinger(series: pd.Series, window=20, std=2):
    sma   = series.rolling(window, min_periods=1).mean()
    sigma = series.rolling(window, min_periods=1).std()
    return sma, sma + std * sigma, sma - std * sigma


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


# ════════════════════════════════════════════════════════════════════════════
# 3.  LONG-TERM FUNDAMENTAL SCORECARD
# ════════════════════════════════════════════════════════════════════════════

SECTOR_ROE_BENCHMARKS = {
    "bank": 0.12, "financial": 0.12, "insurance": 0.12,
    "technology": 0.20, "software": 0.20, "it ": 0.18,
    "energy": 0.10, "oil": 0.10, "power": 0.09,
    "consumer": 0.18, "fmcg": 0.18,
    "auto": 0.14, "automobile": 0.14,
    "pharma": 0.16, "health": 0.16,
    "default": 0.15,
}

def sector_roe_threshold(sector_str: str) -> float:
    s = (sector_str or "").lower()
    for key, thresh in SECTOR_ROE_BENCHMARKS.items():
        if key in s:
            return thresh
    return SECTOR_ROE_BENCHMARKS["default"]


def long_term_score(info: dict) -> dict:
    reasons   = {"quality": [], "valuation": [], "timing": [], "flags": []}
    score     = {"quality": 0, "valuation": 0, "timing": 0}
    sector = info.get("sector", "") or info.get("industry", "") or ""

    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        if rev_growth >= 0.20:
            score["quality"] += 14
            reasons["quality"].append(f"Excellent revenue growth ({rev_growth*100:.1f}% YoY).")
        elif rev_growth >= 0.12:
            score["quality"] += 10
            reasons["quality"].append(f"Healthy revenue growth ({rev_growth*100:.1f}% YoY).")
        elif rev_growth >= 0.05:
            score["quality"] += 5
            reasons["quality"].append(f"Moderate revenue growth ({rev_growth*100:.1f}% YoY).")
        elif rev_growth < 0:
            reasons["flags"].append(f" Revenue declining ({rev_growth*100:.1f}% YoY).")
        else:
            reasons["quality"].append(f"Slow revenue growth ({rev_growth*100:.1f}% YoY).")
    else:
        reasons["quality"].append("Revenue growth data unavailable.")

    fcf         = info.get("freeCashflow")
    total_rev   = info.get("totalRevenue")
    fcf_margin  = None
    if fcf and total_rev and total_rev > 0:
        fcf_margin = fcf / total_rev
        if fcf_margin >= 0.15:
            score["quality"] += 14
            reasons["quality"].append(f"Strong FCF margin ({fcf_margin*100:.1f}%). Cash-generative business.")
        elif fcf_margin >= 0.08:
            score["quality"] += 9
            reasons["quality"].append(f"Decent FCF margin ({fcf_margin*100:.1f}%).")
        elif fcf_margin >= 0:
            score["quality"] += 4
            reasons["quality"].append(f"Thin but positive FCF margin ({fcf_margin*100:.1f}%).")
        else:
            reasons["flags"].append(f" Negative FCF margin ({fcf_margin*100:.1f}%). Cash burn detected.")
    else:
        reasons["quality"].append("FCF data unavailable.")

    roe       = info.get("returnOnEquity")
    roe_bench = sector_roe_threshold(sector)
    if roe is not None:
        if roe >= roe_bench * 1.33:
            score["quality"] += 12
            reasons["quality"].append(f"Outstanding ROE ({roe*100:.1f}%) — well above sector benchmark.")
        elif roe >= roe_bench:
            score["quality"] += 8
            reasons["quality"].append(f"Good ROE ({roe*100:.1f}%) — meets sector benchmark ({roe_bench*100:.0f}%).")
        elif roe >= roe_bench * 0.67:
            score["quality"] += 4
            reasons["quality"].append(f"Below-benchmark ROE ({roe*100:.1f}%). Sector avg ~{roe_bench*100:.0f}%.")
        else:
            reasons["flags"].append(f" Weak ROE ({roe*100:.1f}%). Sector benchmark is {roe_bench*100:.0f}%.")
    else:
        reasons["quality"].append("ROE data unavailable.")

    eq_growth = info.get("earningsQuarterlyGrowth")
    if eq_growth is not None:
        if eq_growth >= 0.15:
            score["quality"] += 10
            reasons["quality"].append(f"Strong earnings growth ({eq_growth*100:.1f}% YoY).")
        elif eq_growth >= 0.05:
            score["quality"] += 6
            reasons["quality"].append(f"Moderate earnings growth ({eq_growth*100:.1f}% YoY).")
        elif eq_growth < 0:
            reasons["flags"].append(f" Earnings contracting ({eq_growth*100:.1f}% YoY).")
        else:
            score["quality"] += 2
            reasons["quality"].append(f"Flat earnings growth ({eq_growth*100:.1f}% YoY).")

    pe = info.get("trailingPE")
    peg = info.get("trailingPegRatio") or info.get("pegRatio")
    if peg is not None and peg > 0:
        if peg < 1.0:
            score["valuation"] += 12
            reasons["valuation"].append(f"Excellent PEG ({peg:.2f}) — growing faster than its valuation.")
        elif peg < 1.5:
            score["valuation"] += 8
            reasons["valuation"].append(f"Reasonable PEG ({peg:.2f}).")
        elif peg < 2.5:
            score["valuation"] += 4
            reasons["valuation"].append(f"Elevated PEG ({peg:.2f}) — some growth premium priced in.")
        else:
            reasons["flags"].append(f" Very high PEG ({peg:.2f}) — valuation stretched vs growth.")
    elif pe is not None and pe > 0:
        if pe < 15:
            score["valuation"] += 10
            reasons["valuation"].append(f"Attractive PE ({pe:.1f}).")
        elif pe < 25:
            score["valuation"] += 6
            reasons["valuation"].append(f"Reasonable PE ({pe:.1f}).")
        elif pe < 40:
            score["valuation"] += 3
            reasons["valuation"].append(f"Rich PE ({pe:.1f}) — justified only by high growth.")
        else:
            reasons["flags"].append(f" High PE ({pe:.1f}) — demands sustained growth to justify.")

    pm = info.get("profitMargins")
    if pm is not None:
        if pm >= 0.20:
            score["valuation"] += 10
            reasons["valuation"].append(f"Excellent profit margins ({pm*100:.1f}%).")
        elif pm >= 0.12:
            score["valuation"] += 7
            reasons["valuation"].append(f"Good profit margins ({pm*100:.1f}%).")
        elif pm >= 0.06:
            score["valuation"] += 3
            reasons["valuation"].append(f"Thin profit margins ({pm*100:.1f}%).")
        else:
            reasons["flags"].append(f" Very thin margins ({pm*100:.1f}%).")

    de = info.get("debtToEquity")
    if de is not None:
        if de < 30:
            score["valuation"] += 8
            reasons["valuation"].append(f"Conservative leverage (D/E {de:.0f}%).")
        elif de < 70:
            score["valuation"] += 5
            reasons["valuation"].append(f"Moderate leverage (D/E {de:.0f}%).")
        elif de < 120:
            score["valuation"] += 2
            reasons["valuation"].append(f"Elevated leverage (D/E {de:.0f}%).")
        else:
            reasons["flags"].append(f" High leverage (D/E {de:.0f}%). Watch interest coverage.")

    div_yield   = info.get("dividendYield") or 0
    payout_ratio = info.get("payoutRatio") or 0
    if div_yield and div_yield > 0.01:
        if div_yield >= 0.03 and payout_ratio < 0.70:
            reasons["valuation"].append(
                f"Attractive dividend yield ({div_yield*100:.1f}%) with sustainable payout ({payout_ratio*100:.0f}%)."
            )
        elif div_yield >= 0.015:
            reasons["valuation"].append(f"Moderate dividend yield ({div_yield*100:.1f}%).")

    total = score["quality"] + score["valuation"]

    if total >= 65:
        label, color = "Buffett-Grade Compounder", "green"
    elif total >= 45:
        label, color = "Good Quality Business", "teal"
    elif total >= 30:
        label, color = "Average / Watch Closely", "orange"
    else:
        label, color = "Weak Fundamentals", "red"

    return {
        "total_score": total,
        "quality_score": score["quality"],
        "valuation_score": score["valuation"],
        "label": label,
        "color": color,
        "reasons": reasons,
        "pe": pe,
        "peg": peg,
        "roe": roe,
        "rev_growth": rev_growth,
        "fcf_margin": fcf_margin,
        "profit_margin": pm,
        "debt_to_eq": de,
        "div_yield": div_yield,
        "payout_ratio": payout_ratio,
        "sector": sector,
        "earnings_growth": eq_growth,
    }


# ════════════════════════════════════════════════════════════════════════════
# 4.  LONG-TERM SIGNAL ENGINE
# ════════════════════════════════════════════════════════════════════════════

def long_term_signal(
    close: pd.Series,
    avg_price: float,
    fund: dict,
    purchase_date: date | None = None,
) -> dict:
    current_val = close.iloc[-1]
    if isinstance(current_val, (pd.Series, pd.DataFrame)):
        current_val = current_val.iloc[0]
    try:
        current_price = float(current_val)
    except (TypeError, ValueError):
        current_price = 0.0
    if np.isnan(current_price):
        current_price = 0.0

    pnl_pct = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0.0

    holding_years = None
    ltcg_eligible = False
    if purchase_date:
        delta = (date.today() - purchase_date).days
        holding_years  = delta / 365.25
        ltcg_eligible  = delta >= 365

    daily_ret      = close.pct_change().dropna()
    recent         = daily_ret.tail(504) if len(daily_ret) > 504 else daily_ret
    pct_down_days  = float((recent < 0).mean() * 100) if len(recent) else 0.0
    var_95         = float(np.percentile(recent, 5) * 100) if len(recent) else 0.0
    annual_vol     = float(recent.std() * np.sqrt(252) * 100) if len(recent) else 0.0

    sma_200 = float(compute_sma(close, 200).iloc[-1])
    sma_50  = float(compute_sma(close, 50).iloc[-1])
    rsi_val = float(compute_rsi(close).iloc[-1])

    close_1yr = close.tail(252)
    low_52w   = float(close_1yr.min())
    high_52w  = float(close_1yr.max())
    range_pos = (current_price - low_52w) / (high_52w - low_52w) if high_52w > low_52w else 0.5

    monthly_macd_bull, monthly_macd_cross = compute_macd_monthly(close)

    timing_score = 0
    timing_notes = []

    if current_price > sma_200:
        timing_score += 6
        timing_notes.append("Price above 200-DMA — long-term uptrend intact.")
    else:
        timing_notes.append("Price below 200-DMA — long-term trend broken.")

    if monthly_macd_bull:
        timing_score += 7
        if monthly_macd_cross:
            timing_notes.append("Monthly MACD just turned bullish — strong entry signal.")
        else:
            timing_notes.append("Monthly MACD bullish — momentum supportive.")
    else:
        timing_notes.append("Monthly MACD bearish — momentum not yet confirming.")

    if range_pos <= 0.35:
        timing_score += 7
        timing_notes.append(f"Near 52-week low ({range_pos*100:.0f}th percentile) — potential value entry.")
    elif range_pos <= 0.60:
        timing_score += 4
        timing_notes.append(f"Mid-range on 52-week scale — fair entry zone.")
    else:
        timing_notes.append(f"Near 52-week high ({range_pos*100:.0f}th percentile) — wait for pullback to accumulate.")

    combined = fund["total_score"] + timing_score
    has_red_flags = len(fund["reasons"]["flags"]) > 0

    if has_red_flags and fund["quality_score"] < 20:
        signal = " EXIT / AVOID"
        color  = "red"
        reason = (
            "Multiple fundamental red flags detected: "
            + "; ".join(fund["reasons"]["flags"])
            + " — business quality is too weak for long-term holding."
        )
    elif fund["quality_score"] >= 30 and timing_score >= 13:
        signal = " STRONG ACCUMULATE"
        color  = "green"
        reason = (
            "High-quality business at a good entry point. "
            + (timing_notes[0] if timing_notes else "")
            + " Add meaningfully to position."
        )
    elif fund["quality_score"] >= 24 and timing_score >= 8:
        signal = " ACCUMULATE"
        color  = "green"
        reason = (
            "Good business, reasonable entry. "
            + " ".join(timing_notes[:2])
        )
    elif fund["quality_score"] >= 30 and timing_score < 8:
        signal = " HOLD — WAIT FOR BETTER ENTRY"
        color  = "gray"
        reason = (
            "Excellent business but technicals suggest waiting for a pullback. "
            + " ".join(timing_notes[:2])
            + " Set a buy target near the 200-DMA or 52-week mid-range."
        )
    elif fund["quality_score"] >= 18 and pnl_pct < -25 and not has_red_flags:
        signal = " ADD ON WEAKNESS"
        color  = "orange"
        reason = (
            f"Down {abs(pnl_pct):.1f}% from your cost — if the business thesis is intact, "
            "this is a buying opportunity, not a reason to sell. Review the latest earnings."
        )
    elif has_red_flags and fund["quality_score"] >= 18 and pnl_pct > 40:
        signal = " PARTIAL PROFIT BOOKING"
        color  = "orange"
        reason = (
            f"Up {pnl_pct:.1f}% with emerging fundamental concerns. "
            + "; ".join(fund["reasons"]["flags"])
            + " Consider trimming 25–30% of position."
            + (" LTCG-eligible — tax-efficient to book now." if ltcg_eligible else "")
        )
    elif fund["quality_score"] < 18 and pnl_pct < -20:
        signal = " REVIEW & EXIT"
        color  = "red"
        reason = (
            f"Weak fundamentals AND down {abs(pnl_pct):.1f}%. "
            "Business quality doesn't justify holding through further losses."
        )
    else:
        signal = " HOLD"
        color  = "gray"
        reason = (
            "Business quality is adequate. "
            + " ".join(timing_notes[:1])
            + " No strong action needed — review on next earnings."
        )

    return {
        "signal":        signal,
        "color":         color,
        "reason":        reason,
        "combined_score": combined,
        "timing_score":  timing_score,
        "timing_notes":  timing_notes,
        "current_price": current_price,
        "pnl_pct":       pnl_pct,
        "holding_years": holding_years,
        "ltcg_eligible": ltcg_eligible,
        "sma_50":        sma_50,
        "sma_200":       sma_200,
        "rsi_val":       rsi_val,
        "range_pos":     range_pos,
        "low_52w":       low_52w,
        "high_52w":      high_52w,
        "monthly_macd_bull": monthly_macd_bull,
        "pct_down_days": pct_down_days,
        "var_95":        var_95,
        "annual_vol":    annual_vol,
    }


# ════════════════════════════════════════════════════════════════════════════
# 5.  FULL ANALYSIS WRAPPER
# ════════════════════════════════════════════════════════════════════════════

def _extract_close(history: pd.DataFrame) -> pd.Series:
    col = history["Close"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    col = col.squeeze()
    if not isinstance(col, pd.Series):
        col = pd.Series(col, dtype=float)
    col = pd.to_numeric(col, errors="coerce").dropna()
    return col


def analyze_stock(
    history: pd.DataFrame,
    symbol: str,
    avg_price: float = 0.0,
    fund: dict | None = None,
    purchase_date: date | None = None,
) -> dict | None:
    if history.empty or len(history) < 30:
        return None

    close = _extract_close(history)
    if len(close) < 30:
        return None

    rsi_series          = compute_rsi(close)
    macd_line, macd_sig, macd_hist = compute_macd(close)
    bb_mid, bb_upper, bb_lower    = compute_bollinger(close)
    sma_50_s  = compute_sma(close, 50)
    sma_200_s = compute_sma(close, 200)

    sig = long_term_signal(close, avg_price, fund or {}, purchase_date)

    return {
        "symbol":        symbol,
        "history":       history,
        "close":         close,
        "sma_50_s":      sma_50_s,
        "sma_200_s":     sma_200_s,
        "rsi_series":    rsi_series,
        "macd_line":     macd_line,
        "macd_signal":   macd_sig,
        "macd_hist":     macd_hist,
        "bb_mid":        bb_mid,
        "bb_upper":      bb_upper,
        "bb_lower":      bb_lower,
        **sig,
    }


# ════════════════════════════════════════════════════════════════════════════
# 6.  AI SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def get_ai_summary(symbol: str, analysis: dict, fund: dict) -> str:
    holding_str = (
        f"{analysis['holding_years']:.1f} years"
        if analysis.get("holding_years") else "Unknown"
    )
    ltcg_str = "Yes (LTCG-eligible)" if analysis.get("ltcg_eligible") else "Not yet (STCG applicable)"

    prompt = f"""
You are a senior Indian equity analyst advising a long-term investor (3–7 year horizon).
Analyse this stock and give a grounded, honest view suitable for long-term wealth creation.

Symbol: {symbol}
Current price: ₹{analysis['current_price']:.2f}
P&L: {analysis['pnl_pct']:.1f}%
Holding period: {holding_str}
LTCG eligible: {ltcg_str}

── Long-Term Fundamental Scorecard ──
Label: {fund.get('label')}
Total score: {fund.get('total_score')}/80
Quality score: {fund.get('quality_score')}/50
Valuation score: {fund.get('valuation_score')}/30
Revenue growth: {(fund.get('rev_growth') or 0)*100:.1f}%
FCF margin: {(fund.get('fcf_margin') or 0)*100:.1f}%
ROE: {(fund.get('roe') or 0)*100:.1f}%  (sector: {fund.get('sector','N/A')})
PEG ratio: {fund.get('peg') or 'N/A'}
Earnings growth (QoQ): {(fund.get('earnings_growth') or 0)*100:.1f}%
Profit margin: {(fund.get('profit_margin') or 0)*100:.1f}%
Debt/Equity: {fund.get('debt_to_eq') or 'N/A'}
Dividend yield: {(fund.get('div_yield') or 0)*100:.1f}%
Red flags: {'; '.join(fund.get('reasons',{}).get('flags',[])) or 'None'}

── Timing Signals ──
Signal: {analysis['signal']}
Combined score: {analysis['combined_score']}/100
Monthly MACD bullish: {analysis['monthly_macd_bull']}
Price vs 200-DMA: {'above' if analysis['current_price'] > analysis['sma_200'] else 'below'}
52-week position: {analysis['range_pos']*100:.0f}th percentile
Annual volatility: {analysis['annual_vol']:.1f}%

Write a concise analysis (max 200 words):
1. Business quality in 2–3 sentences — focus on FCF, revenue growth, ROE. Be specific.
2. Valuation in 1–2 sentences. Use PEG if available.
3. Clear long-term action recommendation (Accumulate / Hold / Partial profit / Exit) with specific reasoning.
4. One tax note if LTCG-eligible and action is to sell/trim.
Tone: direct, analytical, no hype. Speak like a fiduciary advisor.
"""
    try:
        return call_ai(prompt).strip()
    except Exception as e:
        return f"AI unavailable: {e}"


# ════════════════════════════════════════════════════════════════════════════
# 7.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_portfolio(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=22, sheet_name="Equity")
    df = df.dropna(subset=["Symbol"])
    df = df[df["Symbol"] != "Total"]
    return df


def _flatten_yf(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.loc[:, ~data.columns.duplicated()]
    return data


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_history(ticker: str, period: str = "max") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    data = _flatten_yf(data)
    if data.empty and ticker.endswith(".NS"):
        data = yf.download(ticker.replace(".NS", ".BO"), period=period, interval="1d", progress=False, auto_adjust=True)
        data = _flatten_yf(data)
    return data


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def yahoo_ticker(symbol: str) -> str:
    if "-" in symbol and not symbol.endswith("-N"):
        symbol = symbol.split("-")[0]
    return f"{symbol}.NS"


# ════════════════════════════════════════════════════════════════════════════
# 8.  RENDERING HELPERS
# ════════════════════════════════════════════════════════════════════════════

SIGNAL_COLORS = {
    "green":  "#1a9e6a",
    "teal":   "#0b7b6a",
    "orange": "#e07b00",
    "red":    "#c0392b",
    "gray":   "#606060",
}

CHART_RANGE_OPTIONS = {
    "1D":  1,
    "1W":  7,
    "1M":  30,
    "6M":  182,
    "1Y":  365,
    "5Y":  1825,
    "Max": None,
}


def score_bar(label: str, value: int, max_val: int, color: str = "#1a9e6a"):
    pct = int(value / max_val * 100)
    st.markdown(
        f"""
        <div style="margin-bottom:6px">
          <div style="display:flex;justify-content:space-between;font-size:12px;color:#aaa">
            <span>{label}</span><span>{value}/{max_val}</span>
          </div>
          <div style="background:#333;border-radius:4px;height:6px;margin-top:2px">
            <div style="width:{pct}%;background:{color};height:6px;border-radius:4px"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stock_card(
    analysis: dict,
    avg_price: float = 0.0,
    fund: dict | None = None,
    purchase_date: date | None = None,
    ai_enabled: bool = True,
):
    fund = fund or {}
    symbol  = analysis["symbol"]
    history = analysis["history"]
    color   = SIGNAL_COLORS.get(analysis["color"], "#606060")

    selected_range = st.radio(
        " Chart range",
        options=list(CHART_RANGE_OPTIONS.keys()),
        index=5,
        horizontal=True,
        key=f"range_{symbol}",
    )

    days = CHART_RANGE_OPTIONS[selected_range]
    if days is not None:
        chart_history = history.tail(days)
    else:
        chart_history = history

    def _slice(series: pd.Series) -> pd.Series:
        return series.loc[series.index.isin(chart_history.index)]

    chart_close    = _slice(analysis["close"])
    chart_sma50    = _slice(analysis["sma_50_s"])
    chart_sma200   = _slice(analysis["sma_200_s"])
    chart_bb_upper = _slice(analysis["bb_upper"])
    chart_bb_lower = _slice(analysis["bb_lower"])
    chart_macd     = _slice(analysis["macd_line"])
    chart_signal   = _slice(analysis["macd_signal"])
    chart_hist_s   = _slice(analysis["macd_hist"])
    chart_rsi      = _slice(analysis["rsi_series"])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=("Price · SMAs · Bollinger Bands", "MACD (daily)", "RSI (14)"),
    )

    fig.add_trace(go.Scatter(x=chart_history.index, y=chart_close, name="Price",
                             line=dict(color="#2E86C1", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_history.index, y=chart_sma50, name="50-DMA",
                             line=dict(color="orange", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_history.index, y=chart_sma200, name="200-DMA",
                             line=dict(color="purple", dash="dot", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_history.index, y=chart_bb_upper, name="Upper BB",
                             line=dict(color="gray", width=0.8), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_history.index, y=chart_bb_lower, name="Lower BB",
                             line=dict(color="gray", width=0.8),
                             fill="tonexty", fillcolor="rgba(128,128,128,0.08)",
                             showlegend=False), row=1, col=1)
    if avg_price > 0:
        fig.add_hline(
            y=avg_price, line_dash="dash",
            line_color="green" if analysis["pnl_pct"] > 0 else "red",
            annotation_text=f"Your avg ₹{avg_price:.2f}", row=1, col=1,
        )

    fig.add_trace(go.Scatter(x=chart_history.index, y=chart_macd, name="MACD",
                             line=dict(color="cyan", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart_history.index, y=chart_signal, name="Signal",
                             line=dict(color="magenta", width=1)), row=2, col=1)
    colors_hist = np.where(chart_hist_s >= 0, "#1a9e6a", "#c0392b")
    fig.add_trace(go.Bar(x=chart_history.index, y=chart_hist_s,
                         name="Histogram", marker_color=colors_hist), row=2, col=1)

    fig.add_trace(go.Scatter(x=chart_history.index, y=chart_rsi, name="RSI",
                             line=dict(color="#f39c12", width=1)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",     line_width=0.8, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#1a9e6a",  line_width=0.8, row=3, col=1)

    fig.update_layout(
        height=580, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="#111", paper_bgcolor="#111",
    )
    fig.update_xaxes(showgrid=False, gridcolor="#222")
    fig.update_yaxes(showgrid=True, gridcolor="#1e1e1e")

    left, right = st.columns([1, 2])

    with left:
        st.subheader(f" {symbol}")
        st.metric("Current Price", f"₹{analysis['current_price']:.2f}")
        if avg_price > 0:
            pnl_color = "green" if analysis["pnl_pct"] > 0 else "red"
            st.markdown(
                f"**P&L:** <span style='color:{pnl_color};font-size:22px;font-weight:700'>"
                f"{analysis['pnl_pct']:+.2f}%</span>",
                unsafe_allow_html=True,
            )

        if analysis.get("holding_years") is not None:
            hy = analysis["holding_years"]
            ltcg = " LTCG-eligible" if analysis["ltcg_eligible"] else "⏳ STCG (< 1 yr)"
            st.markdown(f"**Held:** {hy:.1f} yrs &nbsp;|&nbsp; {ltcg}")

        st.markdown(
            f"<div style='background:#1a1a1a;padding:14px;border-radius:10px;"
            f"border-left:5px solid {color};margin:12px 0'>"
            f"<b style='color:{color};font-size:17px'>{analysis['signal']}</b><br/>"
            f"<span style='color:#ccc;font-size:13px'>{analysis['reason']}</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Long-Term Score")
        score_bar("Business Quality", fund.get("quality_score", 0), 50, "#1a9e6a")
        score_bar("Valuation",        fund.get("valuation_score", 0), 30, "#2980b9")
        score_bar("Entry Timing",     analysis.get("timing_score", 0), 20, "#e07b00")
        st.markdown(
            f"<div style='text-align:right;font-size:13px;color:#aaa'>"
            f"Total: <b style='color:#fff'>{analysis.get('combined_score',0)}/100</b></div>",
            unsafe_allow_html=True,
        )

        with st.expander(" Timing signals"):
            for note in analysis.get("timing_notes", []):
                icon = "" if any(w in note for w in ["bullish", "above", "low", "fair", "mid"]) else ""
                st.markdown(f"{icon} {note}")
            st.markdown(f"**52-wk range position:** {analysis['range_pos']*100:.0f}th percentile")
            st.markdown(f"**RSI (14-day):** {analysis['rsi_val']:.1f}")
            st.markdown(f"**Annual volatility:** {analysis['annual_vol']:.1f}%")

        with st.expander(" Risk metrics (2-yr)"):
            st.markdown(f" Down days: **{analysis['pct_down_days']:.1f}%**")
            st.markdown(f" 95% Daily VaR: **{analysis['var_95']:.2f}%**")
            st.markdown(f" Annual volatility: **{analysis['annual_vol']:.1f}%**")

        if fund:
            with st.expander(" Full fundamental scorecard"):
                fc = SIGNAL_COLORS.get(fund.get("color", "gray"), "#606060")
                st.markdown(
                    f"<div style='background:#1a1a1a;padding:10px;border-radius:8px;"
                    f"border-left:4px solid {fc}'>"
                    f"<b style='color:{fc}'>{fund.get('label')}</b></div>",
                    unsafe_allow_html=True,
                )
                all_reasons = (
                    fund.get("reasons", {}).get("quality", [])
                    + fund.get("reasons", {}).get("valuation", [])
                    + fund.get("reasons", {}).get("flags", [])
                )
                for r in all_reasons:
                    st.markdown(f"• {r}")
                rows = []
                def _fmt(v, pct=False, prefix=""):
                    if v is None: return "N/A"
                    if pct: return f"{prefix}{v*100:.1f}%"
                    return f"{prefix}{v:.2f}"
                rows.append({"Metric": "Revenue Growth",  "Value": _fmt(fund.get("rev_growth"), pct=True)})
                rows.append({"Metric": "FCF Margin",      "Value": _fmt(fund.get("fcf_margin"), pct=True)})
                rows.append({"Metric": "ROE",             "Value": _fmt(fund.get("roe"), pct=True)})
                rows.append({"Metric": "PEG Ratio",       "Value": _fmt(fund.get("peg"))})
                rows.append({"Metric": "PE Ratio",        "Value": _fmt(fund.get("pe"))})
                rows.append({"Metric": "Profit Margin",   "Value": _fmt(fund.get("profit_margin"), pct=True)})
                rows.append({"Metric": "Debt/Equity",     "Value": _fmt(fund.get("debt_to_eq"))})
                rows.append({"Metric": "Dividend Yield",  "Value": _fmt(fund.get("div_yield"), pct=True)})
                rows.append({"Metric": "Earnings Growth", "Value": _fmt(fund.get("earnings_growth"), pct=True)})
                st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)

    with right:
        st.plotly_chart(fig, use_container_width=True)

    if ai_enabled:
        with st.expander(" AI Long-Term Analysis"):
            if st.button("Generate AI Analysis", key=f"ai_{symbol}"):
                with st.spinner("Generating deep analysis..."):
                    txt = get_ai_summary(symbol, analysis, fund)
                st.markdown(txt)


# ════════════════════════════════════════════════════════════════════════════
# 9.  WATCHLIST / CATEGORIES
# ════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    "Technology / IT":    ["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS","LTTS.NS"],
    "Banking & Finance":  ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS","BAJFINANCE.NS","BAJAJFINSV.NS","CHOLAFIN.NS","MUTHOOTFIN.NS","PAYTM.NS"],
    "Auto / Mfg":         ["TATAMOTORS.NS","MARUTI.NS","M&M.NS","BAJAJAUTO.NS","EICHERMOT.NS","HEROMOTOCO.NS","TVSMOTOR.NS","ASHOKLEY.NS","BOSCHLTD.NS","CUMMINSIND.NS"],
    "Energy / Oil & Gas": ["RELIANCE.NS","ONGC.NS","NTPC.NS","POWERGRID.NS","COALINDIA.NS","IOC.NS","BPCL.NS","TATAPOWER.NS","ADANIGREEN.NS","GAIL.NS"],
    "FMCG / Consumer":    ["ITC.NS","HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","GODREJCP.NS","MARICO.NS","COLPAL.NS","UBL.NS","MCDOWELL-N.NS"],
}
TOP_50 = [s for cat in CATEGORIES.values() for s in cat]


# ════════════════════════════════════════════════════════════════════════════
# 10.  PORTFOLIO CONCENTRATION CHART
# ════════════════════════════════════════════════════════════════════════════

def render_concentration_chart(rows: list[dict]):
    sector_map = {}
    for cat, tickers in CATEGORIES.items():
        for t in tickers:
            sym = t.replace(".NS", "").replace(".BO", "")
            sector_map[sym] = cat

    sector_val: dict[str, float] = {}
    for r in rows:
        sym    = r["Symbol"]
        raw_val = r["Current Value (₹)"]
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            val = 0.0
        if np.isnan(val):
            val = 0.0
        sector = sector_map.get(sym, "Other")
        sector_val[sector] = sector_val.get(sector, 0) + val

    total = sum(sector_val.values())
    if total == 0:
        st.info("Sector allocation unavailable — no valid current prices fetched yet.")
        return

    labels = list(sector_val.keys())
    values = [sector_val[l] / total * 100 for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.45, textinfo="label+percent",
        marker=dict(colors=["#1a9e6a","#2980b9","#e07b00","#8e44ad","#c0392b","#606060"]),
    ))
    fig.update_layout(
        title="Portfolio Sector Allocation",
        height=360, margin=dict(l=10,r=10,t=40,b=10),
        paper_bgcolor="#111", plot_bgcolor="#111",
        font_color="#ccc",
        showlegend=True,
    )

    top5_val = sum(sorted([r["Current Value (₹)"] for r in rows], reverse=True)[:5])
    top5_pct = top5_val / total * 100

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Concentration Risk")
        st.metric("Top 5 holdings", f"{top5_pct:.1f}% of portfolio")
        if top5_pct > 60:
            st.warning("High concentration — top 5 stocks dominate your portfolio.")
        elif top5_pct > 45:
            st.info("Moderate concentration — consider diversifying further.")
        else:
            st.success("Well diversified across top holdings.")

        st.markdown("#### Sector breakdown")
        for label, pct in sorted(zip(labels, values), key=lambda x: -x[1]):
            st.markdown(f"**{label}**: {pct:.1f}%")


# ════════════════════════════════════════════════════════════════════════════
# 11.  COMBINED PORTFOLIO OVERVIEW CHART
# ════════════════════════════════════════════════════════════════════════════

def render_combined_overview(
    stock_invested: float,
    stock_value: float,
    mf_invested: float,
    mf_value: float,
):
    """Renders a combined portfolio summary with donut + bar charts."""
    total_invested = stock_invested + mf_invested
    total_value    = stock_value + mf_value
    total_pnl      = total_value - total_invested
    total_pnl_pct  = total_pnl / total_invested * 100 if total_invested > 0 else 0

    # ── Top KPI row ───────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(" Total Invested",   f"₹{total_invested:,.0f}")
    k2.metric(" Total Value",      f"₹{total_value:,.0f}")
    k3.metric(" Total P&L",        f"₹{total_pnl:,.0f}", f"{total_pnl_pct:+.2f}%")
    stock_pct = stock_value / total_value * 100 if total_value > 0 else 0
    mf_pct    = mf_value    / total_value * 100 if total_value > 0 else 0
    k4.metric(" Stocks / MF Split", f"{stock_pct:.0f}% / {mf_pct:.0f}%")

    st.divider()

    # ── Two side-by-side charts ───────────────────────────────────────────
    col_left, col_right = st.columns(2)

    # Donut — allocation by current value
    with col_left:
        fig_donut = go.Figure(go.Pie(
            labels=["Stocks", "Mutual Funds"],
            values=[stock_value, mf_value],
            hole=0.55,
            marker=dict(colors=["#2E86C1", "#1a9e6a"]),
            textinfo="label+percent",
        ))
        fig_donut.update_layout(
            title="Current Value Allocation",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="#111", font_color="#ccc", showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Grouped bar — invested vs current value
    with col_right:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Invested",
            x=["Stocks", "Mutual Funds", "Total"],
            y=[stock_invested, mf_invested, total_invested],
            marker_color="#555",
        ))
        fig_bar.add_trace(go.Bar(
            name="Current Value",
            x=["Stocks", "Mutual Funds", "Total"],
            y=[stock_value, mf_value, total_value],
            marker_color=["#2E86C1", "#1a9e6a", "#f39c12"],
        ))
        fig_bar.update_layout(
            barmode="group",
            title="Invested vs Current Value",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="#111", plot_bgcolor="#111",
            font_color="#ccc",
            legend=dict(orientation="h", y=1.1),
        )
        fig_bar.update_yaxes(showgrid=True, gridcolor="#1e1e1e")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── P&L summary table ─────────────────────────────────────────────────
    stock_pnl     = stock_value - stock_invested
    stock_pnl_pct = stock_pnl / stock_invested * 100 if stock_invested > 0 else 0
    mf_pnl        = mf_value - mf_invested
    mf_pnl_pct    = mf_pnl / mf_invested * 100 if mf_invested > 0 else 0

    summary_data = {
        "Category":       [" Stocks", " Mutual Funds", " Total Portfolio"],
        "Invested (₹)":   [f"₹{stock_invested:,.2f}", f"₹{mf_invested:,.2f}", f"₹{total_invested:,.2f}"],
        "Current (₹)":    [f"₹{stock_value:,.2f}",    f"₹{mf_value:,.2f}",    f"₹{total_value:,.2f}"],
        "P&L (₹)":        [f"₹{stock_pnl:,.2f}",      f"₹{mf_pnl:,.2f}",      f"₹{total_pnl:,.2f}"],
        "P&L (%)":        [f"{stock_pnl_pct:+.2f}%",   f"{mf_pnl_pct:+.2f}%",  f"{total_pnl_pct:+.2f}%"],
    }
    st.dataframe(pd.DataFrame(summary_data).set_index("Category"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# 12.  APP
# ════════════════════════════════════════════════════════════════════════════

st.title(" Nivetha's Portfolio Tracker")
st.caption("Long-term investor edition — fundamental-first signal engine")

excel_path = r"C:\Users\Aravind\control\Porfolio\holdings-UB7034.xlsx"

try:
    portfolio = load_portfolio(excel_path)
except Exception as e:
    st.error(f"Could not load portfolio file: {e}")
    st.stop()

try:
    mf_df = pd.read_excel(excel_path, header=22, sheet_name="Mutual Funds")
    mf_df = mf_df.dropna(subset=["Symbol"])
    mf_df = mf_df[mf_df["Symbol"] != "Total"]
except Exception:
    mf_df = pd.DataFrame()

def safe_float(val):
    if pd.isna(val) or val == "" or str(val).strip() == "-":
        return 0.0
    try:
        if isinstance(val, str):
            val = val.replace(",", "")
        return float(val)
    except ValueError:
        return 0.0

# ── Pre-compute MF totals (needed for combined overview tab) ──────────────
mf_rows          = []
mf_total_invested = 0.0
mf_total_value   = 0.0

if not mf_df.empty:
    for _, row in mf_df.iterrows():
        sym        = str(row["Symbol"]).strip()
        qty        = safe_float(row.get("Quantity Available", 0))
        avg_price  = safe_float(row.get("Average Price", 0))
        prev_close = safe_float(row.get("Previous Closing Price", 0))
        pnl_val    = safe_float(row.get("Unrealized P&L", 0))
        if qty <= 0:
            continue
        invested   = qty * avg_price
        cur_val    = qty * prev_close
        pnl_pct    = safe_float(row.get("Unrealized P&L Pct.", 0))
        mf_total_invested += invested
        mf_total_value    += cur_val
        mf_rows.append({
            "Fund Name":        sym,
            "Units":            qty,
            "Avg Price (₹)":    avg_price,
            "Latest NAV (₹)":   prev_close,
            "Invested (₹)":     invested,
            "Current Value (₹)": cur_val,
            "P&L (₹)":          pnl_val,
            "P&L (%)":          pnl_pct,
        })

# ── TABS ─────────────────────────────────────────────────────────────────
tab_overview, tab_stocks, tab_mf, tab_forecast = st.tabs([
    "My Portfolio",
    "Stock Portfolio",
    "Mutual Funds",
    "Forecast",
])




# ════════════════════════════════════════════════════════════════════════════
# TAB: STOCK PORTFOLIO  (was "My Portfolio")
# ════════════════════════════════════════════════════════════════════════════
with tab_stocks:
    st.header("Nivetha's Stock Holdings")

    with st.expander(" Enter purchase dates for LTCG tracking (optional)"):
        st.caption("Format: YYYY-MM-DD. Leave blank to skip.")
        purchase_dates: dict[str, date | None] = {}
        for _, row in portfolio.iterrows():
            sym = str(row["Symbol"]).strip()
            d_str = st.text_input(f"{sym} purchase date", key=f"pd_{sym}", placeholder="e.g. 2021-06-15")
            try:
                purchase_dates[sym] = date.fromisoformat(d_str) if d_str.strip() else None
            except ValueError:
                purchase_dates[sym] = None

    rows_for_table  = []
    analysis_store  = {}
    fund_store      = {}
    total_invested  = 0.0
    total_value     = 0.0

    for _, row in portfolio.iterrows():
        symbol    = str(row["Symbol"]).strip()
        avg_price = safe_float(row.get("Average Price", 0))
        qty_avail = safe_float(row.get("Quantity Available", 0))
        qty_lt    = safe_float(row.get("Quantity Long Term", 0))
        total_qty = qty_avail + qty_lt
        if total_qty <= 0:
            continue

        ticker  = yahoo_ticker(symbol)
        pd_date = purchase_dates.get(symbol)

        with st.spinner(f"Analysing {symbol}…"):
            try:
                hist = fetch_history(ticker)
                info = fetch_info(ticker)
                fund = long_term_score(info)
                ana  = analyze_stock(hist, symbol, avg_price, fund, pd_date)
                if ana is None:
                    st.warning(f"{symbol}: insufficient history.")
                    continue

                analysis_store[symbol] = ana
                fund_store[symbol]     = fund

                invested = avg_price * total_qty
                cp = ana["current_price"]
                if cp == 0.0 or np.isnan(cp):
                    st.warning(f"{symbol}: could not fetch current price, skipping from totals.")
                    continue
                cur_val  = cp * total_qty
                pnl_val  = cur_val - invested
                pnl_pct  = pnl_val / invested * 100 if invested > 0 else 0

                rows_for_table.append({
                    "Symbol":             symbol,
                    "Qty":                total_qty,
                    "Avg Price (₹)":      avg_price,
                    "Invested (₹)":       invested,
                    "Current Price (₹)":  cp,
                    "Current Value (₹)":  cur_val,
                    "P&L (₹)":            pnl_val,
                    "P&L (%)":            pnl_pct,
                    "Score /100":         ana["combined_score"],
                    "Signal":             ana["signal"],
                    "Fundamental Label":  fund["label"],
                    "LTCG":              "" if ana["ltcg_eligible"] else "—",
                })

                total_invested += invested
                total_value    += cur_val

            except Exception as e:
                st.error(f"{symbol} failed: {e}")

    # Store stock totals in session state for the combined tab
    st.session_state["stock_invested"] = total_invested
    st.session_state["stock_value"]    = total_value

    total_pnl     = total_value - total_invested
    total_pnl_pct = total_pnl / total_invested * 100 if total_invested > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric(" Total Invested",  f"₹{total_invested:,.2f}")
    c2.metric(" Current Value",   f"₹{total_value:,.2f}")
    c3.metric(" Total P&L",       f"₹{total_pnl:,.2f}", f"{total_pnl_pct:+.2f}%")

    st.divider()

    if rows_for_table:
        df_view = pd.DataFrame(rows_for_table).set_index("Symbol")
        st.dataframe(
            df_view.style.format({
                "Qty": "{:,.0f}",
                "Avg Price (₹)": "₹{:,.2f}",
                "Invested (₹)":  "₹{:,.2f}",
                "Current Price (₹)": "₹{:,.2f}",
                "Current Value (₹)": "₹{:,.2f}",
                "P&L (₹)":  "₹{:,.2f}",
                "P&L (%)":  "{:+.2f}%",
                "Score /100": "{:.0f}",
            }).background_gradient(subset=["Score /100"], cmap="RdYlGn"),
            use_container_width=True, height=420,
        )

        st.divider()
        render_concentration_chart(rows_for_table)
        st.divider()

        st.markdown("###  Stock deep-dive")
        selected = st.selectbox("Select a stock", list(analysis_store.keys()))
        if selected:
            render_stock_card(
                analysis_store[selected],
                avg_price=next(
                    r["Avg Price (₹)"] for r in rows_for_table if r["Symbol"] == selected
                ),
                fund=fund_store.get(selected),
                purchase_date=purchase_dates.get(selected),
                ai_enabled=True,
            )

    st.divider()
    st.markdown("###  AI Portfolio Strategist")
    st.caption("Cross-analyses every holding to recommend the top 3 buys and top 3 exits.")

    if st.button("Generate Portfolio Strategy") and rows_for_table:
        with st.spinner("AI is reviewing your full portfolio…"):
            summary = "\n".join(
                f"{r['Symbol']}: P&L {r['P&L (%)']:+.1f}%, Score {r['Score /100']:.0f}/100, "
                f"Signal {r['Signal']}, Fund: {r['Fundamental Label']}, LTCG: {r['LTCG']}"
                for r in rows_for_table
            )
            prompt = f"""
You are a Chief Investment Officer reviewing a long-term Indian equity portfolio.

Portfolio data:
{summary}

Scores are out of 100 (50 business quality + 30 valuation + 20 entry timing).
Signals use a long-term framework — not short-term momentum.

Provide:
1. TOP 3 STOCKS TO ACCUMULATE MORE — with specific reasons based on score, signal, and P&L.
2. TOP 3 STOCKS TO EXIT OR TRIM — with specific reasons. Note LTCG status where relevant.
3. One overall portfolio observation (diversification, risk concentration, etc.)

Format with clear markdown headers. Be direct and specific. No hype.
"""
            try:
                st.success("Analysis complete!")
                st.markdown(call_ai(prompt))
            except Exception as e:
                st.error(f"AI failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB: MY PORTFOLIO  (combined overview — stocks + MF + total)
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header(" My Total Portfolio — Stocks + Mutual Funds")
    st.caption("Combined view of your entire investment portfolio.")

    # Pull stock totals from session state (populated when Stock Portfolio tab runs)
    stock_invested = st.session_state.get("stock_invested", 0.0)
    stock_value    = st.session_state.get("stock_value",    0.0)

    if stock_invested == 0.0 and stock_value == 0.0:
        st.info(
            " **Tip:** Open the **Stock Portfolio** tab first so stock prices are loaded, "
            "then return here to see your combined portfolio overview."
        )
        # Still show what we can from MF alone
        if mf_total_invested > 0:
            st.markdown("#### Mutual Funds (loaded from file)")
            mf_pnl     = mf_total_value - mf_total_invested
            mf_pnl_pct = mf_pnl / mf_total_invested * 100 if mf_total_invested > 0 else 0
            m1, m2, m3 = st.columns(3)
            m1.metric("MF Invested", f"₹{mf_total_invested:,.2f}")
            m2.metric("MF Value",    f"₹{mf_total_value:,.2f}")
            m3.metric("MF P&L",      f"₹{mf_pnl:,.2f}", f"{mf_pnl_pct:+.2f}%")
    else:
        render_combined_overview(
            stock_invested=stock_invested,
            stock_value=stock_value,
            mf_invested=mf_total_invested,
            mf_value=mf_total_value,
        )

        # ── Asset-class P&L trend (sparkline-style bar) ───────────────────
        st.divider()
        st.markdown("###  Holdings Summary")

        # Stock mini-table (quick view)
        if rows_for_table:
            st.markdown("####  Top Stocks by Current Value")
            top_stocks = sorted(rows_for_table, key=lambda x: -x["Current Value (₹)"])[:8]
            st.dataframe(
                pd.DataFrame(top_stocks)[
                    ["Symbol", "Invested (₹)", "Current Value (₹)", "P&L (%)", "Signal"]
                ].set_index("Symbol").style.format({
                    "Invested (₹)":      "₹{:,.2f}",
                    "Current Value (₹)": "₹{:,.2f}",
                    "P&L (%)":           "{:+.2f}%",
                }),
                use_container_width=True,
            )

        # MF mini-table
        if mf_rows:
            st.markdown("####  Mutual Funds")
            st.dataframe(
                pd.DataFrame(mf_rows)[
                    ["Fund Name", "Invested (₹)", "Current Value (₹)", "P&L (%)"]
                ].set_index("Fund Name").style.format({
                    "Invested (₹)":      "₹{:,.2f}",
                    "Current Value (₹)": "₹{:,.2f}",
                    "P&L (%)":           "{:+.2f}%",
                }),
                use_container_width=True,
            )

        # AI overall summary
        st.divider()
        st.markdown("###  AI Overall Portfolio Commentary")
        if st.button("Generate Combined Portfolio Commentary"):
            total_inv = stock_invested + mf_total_invested
            total_val = stock_value + mf_total_value
            total_pnl = total_val - total_inv
            with st.spinner("AI is analysing your full wealth picture…"):
                prompt = f"""
You are a senior wealth manager reviewing a retail Indian investor's total portfolio.

TOTAL PORTFOLIO:
- Total invested: ₹{total_inv:,.0f}
- Current value:  ₹{total_val:,.0f}
- Total P&L:      ₹{total_pnl:,.0f} ({total_pnl/total_inv*100:+.1f}%)
- Stocks value:   ₹{stock_value:,.0f} ({stock_value/total_val*100:.1f}% of portfolio)
- Mutual Funds:   ₹{mf_total_value:,.0f} ({mf_total_value/total_val*100:.1f}% of portfolio)

Provide a 150-word holistic commentary:
1. Overall portfolio health
2. Balance between direct equity and MFs
3. One actionable suggestion to improve the portfolio
Tone: honest, advisor-like, no flattery.
"""
                try:
                    st.markdown(call_ai(prompt))
                except Exception as e:
                    st.error(f"AI failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB: MUTUAL FUNDS
# ════════════════════════════════════════════════════════════════════════════
with tab_mf:
    st.header("Mutual Funds")
    if not mf_rows:
        st.info("No mutual funds found in the portfolio file.")
    else:
        mf_pnl     = mf_total_value - mf_total_invested
        mf_pnl_pct = mf_pnl / mf_total_invested * 100 if mf_total_invested > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric(" Total Invested (MF)", f"₹{mf_total_invested:,.2f}")
        c2.metric(" Current Value (MF)",  f"₹{mf_total_value:,.2f}")
        c3.metric(" Total P&L (MF)",      f"₹{mf_pnl:,.2f}", f"{mf_pnl_pct:+.2f}%")

        st.divider()
        mf_view = pd.DataFrame(mf_rows).set_index("Fund Name")
        st.dataframe(
            mf_view.style.format({
                "Units":             "{:,.3f}",
                "Avg Price (₹)":     "₹{:,.2f}",
                "Latest NAV (₹)":    "₹{:,.2f}",
                "Invested (₹)":      "₹{:,.2f}",
                "Current Value (₹)": "₹{:,.2f}",
                "P&L (₹)":           "₹{:,.2f}",
                "P&L (%)":           "{:+.2f}%",
            }),
            use_container_width=True,
        )



# ════════════════════════════════════════════════════════════════════════════
# TAB: FORECAST
# ════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.header("Portfolio Forecast — Future Value Projections")
    st.caption("Compound growth projections per asset class with customisable assumptions.")

    # ── Default Assumptions (Native) ─────────────────────────────────────
    defaults = {
        "mf_r": 0.12, "fd_r": 0.07, "etf_r": 0.10, "re_r": 0.06, "os_r": 0.075,
        "inf": 0.05, "eur": 106.5, "add": 500000.0, "add_g": 0.05,
    }

    # ── Assumptions UI ────────────────────────────────────────────────────
    st.subheader("Assumptions")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Return Rates (p.a.)**")
        stock_return = st.number_input("Stocks",           0.0, 0.50, 0.14, 0.005, format="%.3f", key="fc_st_r")
        mf_return    = st.number_input("Mutual Funds",     0.0, 0.50, defaults["mf_r"],  0.005, format="%.3f", key="fc_mf_r")
        fd_rate      = st.number_input("Fixed Deposits",   0.0, 0.30, defaults["fd_r"], 0.005, format="%.3f", key="fc_fd_r")
        etf_return   = st.number_input("ETFs (Global)",    0.0, 0.40, defaults["etf_r"],0.005, format="%.3f", key="fc_etf_r")

    with col_b:
        st.markdown("**Macro & Savings**")
        inflation   = st.number_input("India inflation",   0.0, 0.20, defaults["inf"],  0.005, format="%.3f", key="fc_inf")
        annual_add  = st.number_input("Annual Top-up (₹)", 0, 10_000_000, int(defaults["add"]), 50000, key="fc_add")
        add_growth  = st.number_input("Top-up growth p.a.", 0.0, 0.30, defaults["add_g"], 0.005, format="%.3f", key="fc_add_g")
        re_appre    = st.number_input("Real Estate Appr.", 0.0, 0.25, defaults["re_r"], 0.005, format="%.3f", key="fc_re_r")

    with col_c:
        st.markdown("**Current Portfolio Values (₹)**")
        # Pull live values from session state / calcs
        live_stocks = st.session_state.get("stock_value", 0.0)
        live_mf     = mf_total_value

        cur_st  = st.number_input("Stock Portfolio",  0, 200_000_000, int(live_stocks), 10000, key="fc_cur_st")
        cur_mf  = st.number_input("Mutual Funds",     0, 200_000_000, int(live_mf),     10000, key="fc_cur_mf")
        cur_fd  = st.number_input("Fixed Deposits",   0, 100_000_000, 500000,          10000, key="fc_cur_fd")
        cur_re  = st.number_input("Real Estate",      0, 200_000_000, 0,               10000, key="fc_cur_re")
        cur_os  = st.number_input("Other Assets/Savings", 0, 100_000_000, 0,          10000, key="fc_cur_os")

    # ── Projection engine ─────────────────────────────────────────────────
    horizons = [1, 3, 5, 10, 20, 30]

    asset_rates = {
        "Direct Stocks":  (cur_st,  stock_return),
        "Mutual Funds":   (cur_mf,  mf_return),
        "Fixed Deposits": (cur_fd,  fd_rate),
        "Real Estate":    (cur_re,  re_appre),
        "Other Assets":   (cur_os,  0.07),
    }
    total_cur    = sum(v for v, _ in asset_rates.values())
    blended_rate = (sum(r * v for v, r in asset_rates.values()) / total_cur) if total_cur else 0

    def fv_lump(pv, rate, n):
        return pv * (1 + rate) ** n

    def fv_portfolio(pv, rate, n, add, add_g):
        val = pv * (1 + rate) ** n
        for yr in range(1, n + 1):
            val += add * (1 + add_g) ** (yr - 1) * (1 + rate) ** (n - yr)
        return val

    def fmt_inr(v):
        if v >= 1e7:  return f"Rs {v/1e7:.2f} Cr"
        if v >= 1e5:  return f"Rs {v/1e5:.2f} L"
        return f"Rs {v:,.0f}"

    # Nominal table data
    nom_rows = []
    for asset, (pv, rate) in asset_rates.items():
        row = {"Asset Class": asset, "Current Value": pv}
        for n in horizons:
            row[f"{n}Y"] = fv_lump(pv, rate, n)
        nom_rows.append(row)

    # Total row uses blended rate + top-ups
    total_row = {"Asset Class": "TOTAL (with top-ups)", "Current Value": total_cur}
    for n in horizons:
        total_row[f"{n}Y"] = fv_portfolio(total_cur, blended_rate, n, annual_add, add_growth)
    nom_rows.append(total_row)

    # Real table
    real_rows = []
    for row in nom_rows:
        rrow = {"Asset Class": row["Asset Class"], "Current Value": row["Current Value"]}
        for n in horizons:
            rrow[f"{n}Y"] = row[f"{n}Y"] / (1 + inflation) ** n
        real_rows.append(rrow)

    df_nom  = pd.DataFrame(nom_rows).set_index("Asset Class")
    df_real = pd.DataFrame(real_rows).set_index("Asset Class")

    col_rename = {"Current Value": "Current Value"} | {f"{n}Y": f"{n}Y" for n in horizons}

    def highlight_total(row):
        return ["font-weight:bold; background-color:#1a2530" if row.name == "TOTAL (with top-ups)" else "" for _ in row]

    def highlight_total_real(row):
        return ["font-weight:bold; background-color:#152515" if row.name == "TOTAL (with top-ups)" else "" for _ in row]

    st.divider()

    # ── Tables ────────────────────────────────────────────────────────────
    st.subheader("Nominal Projections (INR)")
    st.caption("Future values at nominal prices — not adjusted for inflation.")
    st.dataframe(
        df_nom.style
            .format(lambda v: fmt_inr(v) if isinstance(v, (int, float)) else v)
            .apply(highlight_total, axis=1),
        use_container_width=True, height=265,
    )

    st.subheader(f"Real Projections — Today's Purchasing Power (deflated at {inflation*100:.1f}% CPI)")
    st.dataframe(
        df_real.style
            .format(lambda v: fmt_inr(v) if isinstance(v, (int, float)) else v)
            .apply(highlight_total_real, axis=1),
        use_container_width=True, height=265,
    )

    # ── 30-year growth chart ──────────────────────────────────────────────
    st.divider()
    st.subheader("30-Year Portfolio Growth")

    yrs          = list(range(0, 31))
    vals_nom     = [fv_portfolio(total_cur, blended_rate, y, annual_add, add_growth) for y in yrs]
    vals_real    = [v / (1 + inflation) ** y for y, v in zip(yrs, vals_nom)]
    vals_no_topup = [fv_lump(total_cur, blended_rate, y) for y in yrs]

    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(
        x=yrs, y=vals_nom, name="Nominal (with top-ups)",
        line=dict(color="#2E86C1", width=2.5),
        fill="tozeroy", fillcolor="rgba(46,134,193,0.07)",
    ))
    fig_g.add_trace(go.Scatter(
        x=yrs, y=vals_no_topup, name="Nominal (no top-ups)",
        line=dict(color="#888", width=1.5, dash="dot"),
    ))
    fig_g.add_trace(go.Scatter(
        x=yrs, y=vals_real, name="Real / inflation-adjusted",
        line=dict(color="#1a9e6a", width=2, dash="dash"),
    ))
    for yr in [5, 10, 20, 30]:
        v = fv_portfolio(total_cur, blended_rate, yr, annual_add, add_growth)
        fig_g.add_annotation(
            x=yr, y=v, text=fmt_inr(v),
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor="#555", font=dict(size=10, color="#ccc"),
            bgcolor="#1a1a1a", bordercolor="#444", ax=0, ay=-35,
        )
    fig_g.update_layout(
        height=430, margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="#ccc",
        xaxis=dict(title="Years from now", showgrid=False, dtick=5),
        yaxis=dict(title="Portfolio Value (INR)", showgrid=True, gridcolor="#1e1e1e", tickformat=",.0f"),
        legend=dict(orientation="h", y=1.06),
        hovermode="x unified",
    )
    st.plotly_chart(fig_g, use_container_width=True)

    # ── Stacked bar at milestones ─────────────────────────────────────────
    st.subheader("Asset Allocation at Key Milestones")
    milestones = [0, 1, 5, 10, 20, 30]
    asset_colors = {
        "Mutual Funds":   "#2E86C1",
        "Fixed Deposits": "#1a9e6a",
        "ETFs (Germany)": "#e07b00",
        "Real Estate":    "#8e44ad",
        "Other Savings":  "#c0392b",
    }
    fig_bar = go.Figure()
    for asset, (pv, rate) in asset_rates.items():
        fig_bar.add_trace(go.Bar(
            name=asset,
            x=["Now" if n == 0 else f"Year {n}" for n in milestones],
            y=[fv_lump(pv, rate, n) for n in milestones],
            marker_color=asset_colors.get(asset, "#888"),
        ))
    fig_bar.update_layout(
        barmode="stack", height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="#ccc",
        yaxis=dict(showgrid=True, gridcolor="#1e1e1e", tickformat=",.0f"),
        legend=dict(orientation="h", y=1.06),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Goal Calculator ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Goal Calculator")

    gc1, gc2 = st.columns(2)
    with gc1:
        fire_target = st.number_input("Target corpus (INR)", 1_000_000, 500_000_000,
                                       50_000_000, 1_000_000, key="fc_target")
    with gc2:
        withdrawal_rate = st.number_input("Safe withdrawal rate (% p.a.)", 0.5, 10.0, 3.5, 0.5,
                                           key="fc_swr")

    yr_hit = None
    for yr in range(1, 61):
        if fv_portfolio(total_cur, blended_rate, yr, annual_add, add_growth) >= fire_target:
            yr_hit = yr
            break

    annual_withdrawal = fire_target * withdrawal_rate / 100
    real_target_val   = fire_target / (1 + inflation) ** (yr_hit or 30)

    g1, g2, g3 = st.columns(3)
    if yr_hit:
        g1.metric("Years to target", f"{yr_hit} yrs")
        g2.metric("Annual withdrawal at target", fmt_inr(annual_withdrawal))
        g3.metric("Real value of target", fmt_inr(real_target_val))
        st.success(f"Portfolio projected to reach {fmt_inr(fire_target)} in {yr_hit} years at current assumptions.")
    else:
        g1.metric("Years to target", "> 60 yrs")
        g2.metric("Annual withdrawal at target", fmt_inr(annual_withdrawal))
        g3.metric("Real return over inflation", f"{(blended_rate - inflation)*100:.1f}%")
        st.warning("Target not reached within 60 years. Consider raising annual investment or target return assumptions.")

    # ── AI commentary ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("AI Forecast Commentary")
    if st.button("Generate AI Commentary", key="fc_ai_btn"):
        nom_30  = fv_portfolio(total_cur, blended_rate, 30, annual_add, add_growth)
        real_30 = nom_30 / (1 + inflation) ** 30
        with st.spinner("Generating commentary..."):
            prompt = f"""
You are a senior Indian wealth planner reviewing a long-term portfolio forecast.

CURRENT PORTFOLIO: INR {total_cur:,.0f}
Blended return: {blended_rate*100:.1f}% p.a. | Inflation: {inflation*100:.1f}% p.a.
Annual top-up: INR {annual_add:,} growing at {add_growth*100:.1f}% p.a.

ASSET MIX:
- Mutual Funds:   INR {cur_mf:,}  @ {mf_return*100:.1f}% p.a.
- Fixed Deposits: INR {cur_fd:,}  @ {fd_rate*100:.1f}% p.a.
- ETFs:           INR {cur_etf:,} @ {etf_return*100:.1f}% p.a.
- Real Estate:    INR {cur_re:,}  @ {re_appre*100:.1f}% p.a.
- Other Savings:  INR {cur_os:,}  @ {os_return*100:.1f}% p.a.

PROJECTIONS:
- 5Y  Nominal: INR {fv_portfolio(total_cur, blended_rate, 5, annual_add, add_growth):,.0f}
- 10Y Nominal: INR {fv_portfolio(total_cur, blended_rate, 10, annual_add, add_growth):,.0f}
- 30Y Nominal: INR {nom_30:,.0f}
- 30Y Real:    INR {real_30:,.0f}
Goal target: INR {fire_target:,} — reached in {yr_hit or ">60"} years.

Write a clear 180-word commentary in flowing paragraphs (no bullet points):
1. Whether the blended return assumption is realistic given the asset mix.
2. Whether the 30-year real value represents meaningful wealth in today's terms.
3. One concrete action to improve long-term outcomes (reallocation, SIP increase, etc.).
4. One key risk to watch.
Tone: direct and honest, like a fiduciary advisor. No hype, no flattery.
"""
            try:
                st.markdown(call_ai(prompt))
            except Exception as e:
                st.error(f"AI unavailable: {e}")