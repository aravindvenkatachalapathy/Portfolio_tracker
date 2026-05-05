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

st.set_page_config(layout="wide", page_title="Nivetha's Portfolio Tracker", page_icon="📈")


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
# 2.  TECHNICAL INDICATORS  (kept as timing helpers only)
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
    """Returns (macd_line, signal_line, histogram) — daily for charting."""
    if len(series) < slow + sig:
        z = pd.Series([0.0] * len(series), index=series.index)
        return z, z, z
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal, macd - signal


def compute_macd_monthly(series: pd.Series, fast=12, slow=26, sig=9):
    """Monthly MACD — better for long-term trend confirmation."""
    monthly = series.resample("ME").last().dropna()
    if len(monthly) < slow + sig:
        return False, False          # (is_bullish, just_crossed)
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
# 3.  LONG-TERM FUNDAMENTAL SCORECARD  (redesigned)
# ════════════════════════════════════════════════════════════════════════════

SECTOR_ROE_BENCHMARKS = {
    # sector keyword → minimum good ROE
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
    """
    Weighted 3-layer scorecard designed for long-term investors.

    Layer 1 – Business Quality   (max 50 pts)
    Layer 2 – Valuation          (max 30 pts)
    Layer 3 – Entry Timing       (max 20 pts — caller passes technicals)
    """
    reasons   = {"quality": [], "valuation": [], "timing": [], "flags": []}
    score     = {"quality": 0, "valuation": 0, "timing": 0}

    sector = info.get("sector", "") or info.get("industry", "") or ""

    # ── LAYER 1: Business Quality ────────────────────────────────────────
    # 1a. Revenue Growth (trailing annual)
    rev_growth = info.get("revenueGrowth")          # e.g. 0.18 = 18 %
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
            reasons["flags"].append(f"⚠️ Revenue declining ({rev_growth*100:.1f}% YoY).")
        else:
            reasons["quality"].append(f"Slow revenue growth ({rev_growth*100:.1f}% YoY).")
    else:
        reasons["quality"].append("Revenue growth data unavailable.")

    # 1b. Free Cash Flow margin
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
            reasons["flags"].append(f"⚠️ Negative FCF margin ({fcf_margin*100:.1f}%). Cash burn detected.")
    else:
        reasons["quality"].append("FCF data unavailable.")

    # 1c. Sector-adjusted ROE
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
            reasons["flags"].append(f"⚠️ Weak ROE ({roe*100:.1f}%). Sector benchmark is {roe_bench*100:.0f}%.")
    else:
        reasons["quality"].append("ROE data unavailable.")

    # 1d. Earnings growth (quarterly YoY)
    eq_growth = info.get("earningsQuarterlyGrowth")
    if eq_growth is not None:
        if eq_growth >= 0.15:
            score["quality"] += 10
            reasons["quality"].append(f"Strong earnings growth ({eq_growth*100:.1f}% YoY).")
        elif eq_growth >= 0.05:
            score["quality"] += 6
            reasons["quality"].append(f"Moderate earnings growth ({eq_growth*100:.1f}% YoY).")
        elif eq_growth < 0:
            reasons["flags"].append(f"⚠️ Earnings contracting ({eq_growth*100:.1f}% YoY).")
        else:
            score["quality"] += 2
            reasons["quality"].append(f"Flat earnings growth ({eq_growth*100:.1f}% YoY).")

    # ── LAYER 2: Valuation ───────────────────────────────────────────────
    # 2a. PEG Ratio (PE / earnings growth) — smarter than raw PE
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
            reasons["flags"].append(f"⚠️ Very high PEG ({peg:.2f}) — valuation stretched vs growth.")
    elif pe is not None and pe > 0:
        # Fallback to raw PE with context
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
            reasons["flags"].append(f"⚠️ High PE ({pe:.1f}) — demands sustained growth to justify.")

    # 2b. Profit margin
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
            reasons["flags"].append(f"⚠️ Very thin margins ({pm*100:.1f}%).")

    # 2c. Debt/Equity — leverage check
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
            reasons["flags"].append(f"⚠️ High leverage (D/E {de:.0f}%). Watch interest coverage.")

    # 2d. Dividend yield — bonus for income consistency
    div_yield   = info.get("dividendYield") or 0
    payout_ratio = info.get("payoutRatio") or 0
    if div_yield and div_yield > 0.01:
        if div_yield >= 0.03 and payout_ratio < 0.70:
            reasons["valuation"].append(
                f"Attractive dividend yield ({div_yield*100:.1f}%) with sustainable payout ({payout_ratio*100:.0f}%)."
            )
        elif div_yield >= 0.015:
            reasons["valuation"].append(f"Moderate dividend yield ({div_yield*100:.1f}%).")

    # ── SCORES & LABELS ──────────────────────────────────────────────────
    total = score["quality"] + score["valuation"]  # timing added by caller

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
# 4.  LONG-TERM SIGNAL ENGINE  (redesigned)
# ════════════════════════════════════════════════════════════════════════════

def long_term_signal(
    close: pd.Series,
    avg_price: float,
    fund: dict,
    purchase_date: date | None = None,
) -> dict:
    """
    Returns signal dict with keys: signal, color, reason, timing_score,
    holding_years, ltcg_eligible, risk_metrics.
    """
    # Robustly extract current price — close is already a clean Series from _extract_close
    current_val = close.iloc[-1]
    if isinstance(current_val, (pd.Series, pd.DataFrame)):
        current_val = current_val.iloc[0]
    try:
        current_price = float(current_val)
    except (TypeError, ValueError):
        current_price = 0.0
    if np.isnan(current_price):
        current_price = 0.0

    pnl_pct       = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0.0

    # ── holding period ────────────────────────────────────────────────────
    holding_years = None
    ltcg_eligible = False
    if purchase_date:
        delta = (date.today() - purchase_date).days
        holding_years  = delta / 365.25
        ltcg_eligible  = delta >= 365          # India: 1yr for equity LTCG

    # ── risk metrics (2-yr window) ────────────────────────────────────────
    daily_ret      = close.pct_change().dropna()
    recent         = daily_ret.tail(504) if len(daily_ret) > 504 else daily_ret
    pct_down_days  = float((recent < 0).mean() * 100) if len(recent) else 0.0
    var_95         = float(np.percentile(recent, 5) * 100) if len(recent) else 0.0
    annual_vol     = float(recent.std() * np.sqrt(252) * 100) if len(recent) else 0.0

    # ── technical timing signals ──────────────────────────────────────────
    sma_200 = float(compute_sma(close, 200).iloc[-1])
    sma_50  = float(compute_sma(close, 50).iloc[-1])
    rsi_val = float(compute_rsi(close).iloc[-1])

    # 52-week position (0 = at 52-wk low, 1 = at 52-wk high)
    close_1yr = close.tail(252)
    low_52w   = float(close_1yr.min())
    high_52w  = float(close_1yr.max())
    range_pos = (current_price - low_52w) / (high_52w - low_52w) if high_52w > low_52w else 0.5

    monthly_macd_bull, monthly_macd_cross = compute_macd_monthly(close)

    # timing score (max 20)
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

    # combined score
    combined = fund["total_score"] + timing_score   # max = 80 + 20 = 100

    # ── flags from fundamentals ───────────────────────────────────────────
    has_red_flags = len(fund["reasons"]["flags"]) > 0

    # ── final decision  (fundamental-first, technical-for-timing) ────────
    if has_red_flags and fund["quality_score"] < 20:
        signal = "🔴 EXIT / AVOID"
        color  = "red"
        reason = (
            "Multiple fundamental red flags detected: "
            + "; ".join(fund["reasons"]["flags"])
            + " — business quality is too weak for long-term holding."
        )

    elif fund["quality_score"] >= 30 and timing_score >= 13:
        signal = "🟢 STRONG ACCUMULATE"
        color  = "green"
        reason = (
            "High-quality business at a good entry point. "
            + (timing_notes[0] if timing_notes else "")
            + " Add meaningfully to position."
        )

    elif fund["quality_score"] >= 24 and timing_score >= 8:
        signal = "🟢 ACCUMULATE"
        color  = "green"
        reason = (
            "Good business, reasonable entry. "
            + " ".join(timing_notes[:2])
        )

    elif fund["quality_score"] >= 30 and timing_score < 8:
        signal = "⚪ HOLD — WAIT FOR BETTER ENTRY"
        color  = "gray"
        reason = (
            "Excellent business but technicals suggest waiting for a pullback. "
            + " ".join(timing_notes[:2])
            + " Set a buy target near the 200-DMA or 52-week mid-range."
        )

    elif fund["quality_score"] >= 18 and pnl_pct < -25 and not has_red_flags:
        signal = "🟡 ADD ON WEAKNESS"
        color  = "orange"
        reason = (
            f"Down {abs(pnl_pct):.1f}% from your cost — if the business thesis is intact, "
            "this is a buying opportunity, not a reason to sell. Review the latest earnings."
        )

    elif has_red_flags and fund["quality_score"] >= 18 and pnl_pct > 40:
        signal = "🟡 PARTIAL PROFIT BOOKING"
        color  = "orange"
        reason = (
            f"Up {pnl_pct:.1f}% with emerging fundamental concerns. "
            + "; ".join(fund["reasons"]["flags"])
            + " Consider trimming 25–30% of position."
            + (" LTCG-eligible — tax-efficient to book now." if ltcg_eligible else "")
        )

    elif fund["quality_score"] < 18 and pnl_pct < -20:
        signal = "🔴 REVIEW & EXIT"
        color  = "red"
        reason = (
            f"Weak fundamentals AND down {abs(pnl_pct):.1f}%. "
            "Business quality doesn't justify holding through further losses."
        )

    else:
        signal = "⚪ HOLD"
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
    """
    Safely extract the Close column as a clean float Series,
    handling MultiIndex, DataFrame-shaped columns, and NaN rows.
    """
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

    # daily indicators for charts — always computed on FULL history
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
    """
    yfinance ≥0.2.x returns MultiIndex columns like ('Close', 'TCS.NS').
    Flatten to simple single-level columns so history['Close'] is always a Series.
    """
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        # Drop the ticker level — keep only the field level (Close, Open, …)
        data.columns = data.columns.get_level_values(0)
    # If somehow still duplicated (multiple tickers), keep first occurrence
    data = data.loc[:, ~data.columns.duplicated()]
    return data


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_history(ticker: str, period: str = "max") -> pd.DataFrame:
    # Always fetch max history so indicators are accurate;
    # chart view range is sliced separately in render_stock_card.
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
# 8.  RENDERING
# ════════════════════════════════════════════════════════════════════════════

SIGNAL_COLORS = {
    "green":  "#1a9e6a",
    "teal":   "#0b7b6a",
    "orange": "#e07b00",
    "red":    "#c0392b",
    "gray":   "#606060",
}

# ── Time range options ────────────────────────────────────────────────────
# Maps label → number of calendar days to show (None = all available data)
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

    # ── Time range toggle ─────────────────────────────────────────────────
    selected_range = st.radio(
        "📅 Chart range",
        options=list(CHART_RANGE_OPTIONS.keys()),
        index=5,           # default = 5Y
        horizontal=True,
        key=f"range_{symbol}",
    )

    days = CHART_RANGE_OPTIONS[selected_range]
    if days is not None:
        chart_history = history.tail(days)
    else:
        chart_history = history

    # Helper: slice a full-history series down to the chart window
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

    # ── Chart ─────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=("Price · SMAs · Bollinger Bands", "MACD (daily)", "RSI (14)"),
    )

    # Price
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

    # MACD
    fig.add_trace(go.Scatter(x=chart_history.index, y=chart_macd, name="MACD",
                             line=dict(color="cyan", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart_history.index, y=chart_signal, name="Signal",
                             line=dict(color="magenta", width=1)), row=2, col=1)
    colors_hist = np.where(chart_hist_s >= 0, "#1a9e6a", "#c0392b")
    fig.add_trace(go.Bar(x=chart_history.index, y=chart_hist_s,
                         name="Histogram", marker_color=colors_hist), row=2, col=1)

    # RSI
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

    # ── Layout ────────────────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        st.subheader(f"🔹 {symbol}")
        st.metric("Current Price", f"₹{analysis['current_price']:.2f}")
        if avg_price > 0:
            pnl_color = "green" if analysis["pnl_pct"] > 0 else "red"
            st.markdown(
                f"**P&L:** <span style='color:{pnl_color};font-size:22px;font-weight:700'>"
                f"{analysis['pnl_pct']:+.2f}%</span>",
                unsafe_allow_html=True,
            )

        # Holding info
        if analysis.get("holding_years") is not None:
            hy = analysis["holding_years"]
            ltcg = "✅ LTCG-eligible" if analysis["ltcg_eligible"] else "⏳ STCG (< 1 yr)"
            st.markdown(f"**Held:** {hy:.1f} yrs &nbsp;|&nbsp; {ltcg}")

        # Signal banner
        st.markdown(
            f"<div style='background:#1a1a1a;padding:14px;border-radius:10px;"
            f"border-left:5px solid {color};margin:12px 0'>"
            f"<b style='color:{color};font-size:17px'>{analysis['signal']}</b><br/>"
            f"<span style='color:#ccc;font-size:13px'>{analysis['reason']}</span></div>",
            unsafe_allow_html=True,
        )

        # Composite score bars
        st.markdown("#### Long-Term Score")
        score_bar("Business Quality", fund.get("quality_score", 0), 50, "#1a9e6a")
        score_bar("Valuation",        fund.get("valuation_score", 0), 30, "#2980b9")
        score_bar("Entry Timing",     analysis.get("timing_score", 0), 20, "#e07b00")
        st.markdown(
            f"<div style='text-align:right;font-size:13px;color:#aaa'>"
            f"Total: <b style='color:#fff'>{analysis.get('combined_score',0)}/100</b></div>",
            unsafe_allow_html=True,
        )

        # Timing checklist
        with st.expander("📐 Timing signals"):
            for note in analysis.get("timing_notes", []):
                icon = "✅" if any(w in note for w in ["bullish", "above", "low", "fair", "mid"]) else "⚠️"
                st.markdown(f"{icon} {note}")
            st.markdown(f"**52-wk range position:** {analysis['range_pos']*100:.0f}th percentile")
            st.markdown(f"**RSI (14-day):** {analysis['rsi_val']:.1f}")
            st.markdown(f"**Annual volatility:** {analysis['annual_vol']:.1f}%")

        # Risk
        with st.expander("⚠️ Risk metrics (2-yr)"):
            st.markdown(f"📉 Down days: **{analysis['pct_down_days']:.1f}%**")
            st.markdown(f"📊 95% Daily VaR: **{analysis['var_95']:.2f}%**")
            st.markdown(f"🌊 Annual volatility: **{analysis['annual_vol']:.1f}%**")

        # Fundamentals detail
        if fund:
            with st.expander("📋 Full fundamental scorecard"):
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
                # Key numbers table
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

    # AI Summary
    if ai_enabled:
        with st.expander("🤖 AI Long-Term Analysis"):
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
    # sector allocation by current value
    sector_map = {}
    for cat, tickers in CATEGORIES.items():
        for t in tickers:
            sym = t.replace(".NS", "").replace(".BO", "")
            sector_map[sym] = cat

    sector_val: dict[str, float] = {}
    for r in rows:
        sym    = r["Symbol"]
        raw_val = r["Current Value (₹)"]
        # Guard against NaN current values
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
# 11.  APP
# ════════════════════════════════════════════════════════════════════════════

st.title("📈 Nivetha's Portfolio Tracker")
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

tab1, tab_mf, tab2, tab3 = st.tabs(["💼 My Portfolio", "📈 Mutual Funds", "🔭 Top 50 Watch", "🏭 Sector View"])


# ── TAB 1: My Portfolio ───────────────────────────────────────────────────
with tab1:
    st.header("Nivetha's Holdings")

    # Optional purchase date override (per stock)
    with st.expander("📅 Enter purchase dates for LTCG tracking (optional)"):
        st.caption("Format: YYYY-MM-DD. Leave blank to skip.")
        purchase_dates: dict[str, date | None] = {}
        for _, row in portfolio.iterrows():
            sym = str(row["Symbol"]).strip()
            d_str = st.text_input(f"{sym} purchase date", key=f"pd_{sym}", placeholder="e.g. 2021-06-15")
            try:
                purchase_dates[sym] = date.fromisoformat(d_str) if d_str.strip() else None
            except ValueError:
                purchase_dates[sym] = None

    # ── Main portfolio loop ───────────────────────────────────────────────
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

        ticker    = yahoo_ticker(symbol)
        pd_date   = purchase_dates.get(symbol)

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
                # Skip rows where price couldn't be fetched
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
                    "LTCG":              "✅" if ana["ltcg_eligible"] else "—",
                })

                total_invested += invested
                total_value    += cur_val

            except Exception as e:
                st.error(f"{symbol} failed: {e}")

    # ── Summary metrics ───────────────────────────────────────────────────
    total_pnl     = total_value - total_invested
    total_pnl_pct = total_pnl / total_invested * 100 if total_invested > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Total Invested",  f"₹{total_invested:,.2f}")
    c2.metric("💳 Current Value",   f"₹{total_value:,.2f}")
    c3.metric("📈 Total P&L",       f"₹{total_pnl:,.2f}", f"{total_pnl_pct:+.2f}%")

    st.divider()

    # ── Portfolio table ───────────────────────────────────────────────────
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

        # Concentration chart
        st.divider()
        render_concentration_chart(rows_for_table)
        st.divider()

        # ── Stock deep-dive ───────────────────────────────────────────────
        st.markdown("### 🔍 Stock deep-dive")
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

    # ── AI Portfolio Strategist ───────────────────────────────────────────
    st.divider()
    st.markdown("### 🤖 AI Portfolio Strategist")
    st.caption("Cross-analyses every holding to recommend the top 3 buys and top 3 exits for your long-term portfolio.")

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


# ── TAB MF: Mutual Funds ──────────────────────────────────────────────────
with tab_mf:
    st.header("Mutual Funds")
    if mf_df.empty:
        st.info("No mutual funds found in the portfolio file.")
    else:
        mf_rows = []
        mf_total_invested = 0.0
        mf_total_value = 0.0

        for _, row in mf_df.iterrows():
            sym = str(row["Symbol"]).strip()
            qty = safe_float(row.get("Quantity Available", 0))
            avg_price = safe_float(row.get("Average Price", 0))
            prev_close = safe_float(row.get("Previous Closing Price", 0))
            pnl_val = safe_float(row.get("Unrealized P&L", 0))

            if qty <= 0:
                continue

            invested = qty * avg_price
            cur_val = qty * prev_close
            pnl_pct = safe_float(row.get("Unrealized P&L Pct.", 0))

            mf_total_invested += invested
            mf_total_value += cur_val

            mf_rows.append({
                "Fund Name": sym,
                "Units": qty,
                "Avg Price (₹)": avg_price,
                "Latest NAV (₹)": prev_close,
                "Invested (₹)": invested,
                "Current Value (₹)": cur_val,
                "P&L (₹)": pnl_val,
                "P&L (%)": pnl_pct
            })

        mf_total_pnl = mf_total_value - mf_total_invested
        mf_total_pnl_pct = (mf_total_pnl / mf_total_invested * 100) if mf_total_invested > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Total Invested (MF)", f"₹{mf_total_invested:,.2f}")
        c2.metric("💳 Current Value (MF)", f"₹{mf_total_value:,.2f}")
        c3.metric("📈 Total P&L (MF)", f"₹{mf_total_pnl:,.2f}", f"{mf_total_pnl_pct:+.2f}%")

        st.divider()
        if mf_rows:
            mf_view = pd.DataFrame(mf_rows).set_index("Fund Name")
            st.dataframe(
                mf_view.style.format({
                    "Units": "{:,.3f}",
                    "Avg Price (₹)": "₹{:,.2f}",
                    "Latest NAV (₹)": "₹{:,.2f}",
                    "Invested (₹)": "₹{:,.2f}",
                    "Current Value (₹)": "₹{:,.2f}",
                    "P&L (₹)": "₹{:,.2f}",
                    "P&L (%)": "{:+.2f}%"
                }),
                use_container_width=True
            )


# ── TAB 2: Top 50 Watch ───────────────────────────────────────────────────
with tab2:
    st.header("Top 50 Watchlist — Long-Term Buy Scanner")
    st.caption("Scans 50 major Indian stocks for high fundamental scores + good entry timing.")

    min_score = st.slider("Minimum combined score to show", 40, 80, 55)

    if st.button("Scan Top 50"):
        prog  = st.progress(0)
        hits  = []

        for i, ticker in enumerate(TOP_50):
            try:
                hist = fetch_history(ticker)
                info = fetch_info(ticker)
                fund = long_term_score(info)
                ana  = analyze_stock(hist, ticker.replace(".NS",""), 0.0, fund)
                if ana and ana["combined_score"] >= min_score:
                    hits.append((ana, fund))
            except Exception:
                pass
            prog.progress((i + 1) / len(TOP_50))

        prog.empty()

        if hits:
            hits.sort(key=lambda x: -x[0]["combined_score"])
            st.success(f"Found {len(hits)} stocks scoring ≥ {min_score}/100.")
            for ana, fund in hits:
                st.divider()
                render_stock_card(ana, fund=fund, ai_enabled=True)
        else:
            st.info(f"No stocks scored ≥ {min_score} right now. Try lowering the threshold.")


# ── TAB 3: Sector View ────────────────────────────────────────────────────
with tab3:
    st.header("Sector-Wise Analysis")

    chosen_sector = st.selectbox("Select Sector", list(CATEGORIES.keys()))
    st.markdown(f"### Top 10 in {chosen_sector}")

    for ticker in CATEGORIES[chosen_sector]:
        symbol = ticker.replace(".NS", "")
        st.divider()
        with st.spinner(f"Fetching {symbol}…"):
            try:
                hist = fetch_history(ticker)
                info = fetch_info(ticker)
                fund = long_term_score(info)
                ana  = analyze_stock(hist, symbol, 0.0, fund)
                if ana:
                    render_stock_card(ana, fund=fund, ai_enabled=True)
                else:
                    st.warning(f"{symbol}: insufficient data.")
            except Exception as e:
                st.error(f"{symbol} error: {e}")