# backtest.py
# Signal replay backtesting engine for technical signals

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Literal


def compute_signal_series(hist: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """
    Compute trading signals for each date based on technical indicators.
    Uses rolling indicators only (no fundamentals — unavailable historically).

    Returns: pd.Series indexed by date with signal strings:
        'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    """
    if len(hist) < lookback + 20:
        return pd.Series(['HOLD'] * len(hist), index=hist.index)

    close = hist['Close']

    # Compute indicators
    sma20 = close.rolling(20, min_periods=1).mean()
    sma50 = close.rolling(50, min_periods=1).mean()
    sma200 = close.rolling(200, min_periods=1).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # Bollinger Bands
    bb_middle = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std

    signals = []
    for i in range(len(hist)):
        if i < lookback:
            signals.append('HOLD')
            continue

        score = 0

        # Trend component (0-8 pts)
        if close.iloc[i] > sma200.iloc[i]:
            score += 4
        if close.iloc[i] > sma50.iloc[i]:
            score += 2
        if sma50.iloc[i] > sma200.iloc[i]:
            score += 2

        # Momentum component (0-8 pts)
        if rsi.iloc[i] < 30:
            score += 6  # oversold
        elif rsi.iloc[i] < 45:
            score += 3
        elif rsi.iloc[i] > 70:
            score -= 5  # overbought

        # MACD
        if macd_hist.iloc[i] > 0 and macd_hist.iloc[i] > macd_hist.iloc[i-1]:
            score += 4
        elif macd_hist.iloc[i] < 0:
            score -= 3

        # Bollinger component (0-4 pts)
        if close.iloc[i] <= bb_lower.iloc[i]:
            score += 4
        elif close.iloc[i] >= bb_upper.iloc[i]:
            score -= 3

        # Map score to signal
        if score >= 15:
            signals.append('STRONG_BUY')
        elif score >= 10:
            signals.append('BUY')
        elif score <= -10:
            signals.append('STRONG_SELL')
        elif score <= -5:
            signals.append('SELL')
        else:
            signals.append('HOLD')

    return pd.Series(signals, index=hist.index)


def run_backtest(
    hist: pd.DataFrame,
    ticker: str,
    initial_capital: float = 100_000.0,
    date_range: Literal["1Y", "3Y", "5Y", "Max"] = "3Y"
) -> dict:
    """
    Simulate trading based on signal series.

    Returns dict with: equity_curve, trades, metrics
    """
    # Slice to date range
    if date_range == "1Y":
        lookback_days = 252
    elif date_range == "3Y":
        lookback_days = 252 * 3
    elif date_range == "5Y":
        lookback_days = 252 * 5
    else:  # Max
        lookback_days = len(hist)

    hist_slice = hist.iloc[-lookback_days:]

    # Generate signals
    signals = compute_signal_series(hist_slice)

    # Simulate
    cash = initial_capital
    shares = 0
    entry_price = 0
    entry_date = None

    trades = []
    equity_curve = []

    for i, (date, row) in enumerate(hist_slice.iterrows()):
        signal = signals.iloc[i]
        price = row['Close']

        # Buy signals
        if signal in ('BUY', 'STRONG_BUY') and cash > 0 and shares == 0:
            shares = cash / price
            entry_price = price
            entry_date = date
            trades.append({
                'Entry Date': date,
                'Entry Price': price,
                'Type': 'BUY',
                'Signal': signal,
            })
            cash = 0

        # Partial sell
        elif signal == 'SELL' and shares > 0:
            sell_shares = shares * 0.5
            sell_proceeds = sell_shares * price
            pnl = (price - entry_price) * sell_shares
            cash += sell_proceeds
            shares -= sell_shares
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': date,
                'Entry Price': entry_price,
                'Exit Price': price,
                'Shares': sell_shares,
                'Type': 'PARTIAL_SELL',
                'Signal': signal,
                'PnL': pnl,
                'PnL %': (pnl / (entry_price * sell_shares)) * 100 if entry_price > 0 else 0,
            })

        # Full sell
        elif signal == 'STRONG_SELL' and shares > 0:
            sell_proceeds = shares * price
            pnl = (price - entry_price) * shares
            cash += sell_proceeds
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': date,
                'Entry Price': entry_price,
                'Exit Price': price,
                'Shares': shares,
                'Type': 'SELL',
                'Signal': signal,
                'PnL': pnl,
                'PnL %': (pnl / (entry_price * shares)) * 100 if entry_price > 0 else 0,
            })
            shares = 0

        # Portfolio value
        portfolio_value = cash + (shares * price if shares > 0 else 0)
        equity_curve.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'signal': signal,
            'price': price,
        })

    # Force close at end
    if shares > 0:
        final_price = hist_slice.iloc[-1]['Close']
        pnl = (final_price - entry_price) * shares
        trades.append({
            'Entry Date': entry_date,
            'Exit Date': hist_slice.index[-1],
            'Entry Price': entry_price,
            'Exit Price': final_price,
            'Shares': shares,
            'Type': 'EOD_CLOSE',
            'PnL': pnl,
            'PnL %': (pnl / (entry_price * shares)) * 100 if entry_price > 0 else 0,
        })
        cash += shares * final_price

    # Compute metrics
    metrics = compute_metrics(equity_curve, trades, initial_capital)

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'metrics': metrics,
    }


def compute_metrics(equity_curve: list, trades: list, initial_capital: float) -> dict:
    """
    Compute performance metrics from equity curve and trades.
    """
    if not equity_curve:
        return {
            'total_return_pct': 0,
            'cagr_pct': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'win_rate_pct': 0,
            'num_trades': 0,
            'avg_trade_return_pct': 0,
            'best_trade_pct': 0,
            'worst_trade_pct': 0,
            'buy_hold_return_pct': 0,
        }

    final_value = equity_curve[-1]['portfolio_value']
    start_price = equity_curve[0]['price']
    end_price = equity_curve[-1]['price']

    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100
    buy_hold_return_pct = ((end_price - start_price) / start_price) * 100

    # CAGR
    days = (equity_curve[-1]['date'] - equity_curve[0]['date']).days
    years = max(days / 365, 0.25)  # min 1 quarter
    cagr_pct = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if final_value > 0 else 0

    # Max drawdown
    values = [ec['portfolio_value'] for ec in equity_curve]
    peaks = pd.Series(values).expanding().max()
    drawdowns = (pd.Series(values) - peaks) / peaks
    max_drawdown_pct = drawdowns.min() * 100

    # Sharpe ratio
    daily_returns = pd.Series(values).pct_change().dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Trade stats
    closed_trades = [t for t in trades if 'PnL' in t]
    num_trades = len(closed_trades)
    profitable_trades = [t for t in closed_trades if t.get('PnL', 0) > 0]
    win_rate_pct = (len(profitable_trades) / num_trades * 100) if num_trades > 0 else 0

    trade_returns = [t.get('PnL %', 0) for t in closed_trades]
    avg_trade_return_pct = np.mean(trade_returns) if trade_returns else 0
    best_trade_pct = max(trade_returns) if trade_returns else 0
    worst_trade_pct = min(trade_returns) if trade_returns else 0

    return {
        'total_return_pct': total_return_pct,
        'cagr_pct': cagr_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'win_rate_pct': win_rate_pct,
        'num_trades': num_trades,
        'avg_trade_return_pct': avg_trade_return_pct,
        'best_trade_pct': best_trade_pct,
        'worst_trade_pct': worst_trade_pct,
        'buy_hold_return_pct': buy_hold_return_pct,
    }


def plot_equity_curve(equity_curve: list, trades: list, ticker: str, hist: pd.DataFrame) -> go.Figure:
    """
    Plot equity curve with buy/sell markers and price reference below.
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    # Extract data
    dates = [ec['date'] for ec in equity_curve]
    portfolio_values = [ec['portfolio_value'] for ec in equity_curve]
    prices = [ec['price'] for ec in equity_curve]

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_values,
            name='Portfolio Value',
            line=dict(color='#00D9FF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 255, 0.1)',
        ),
        row=1, col=1,
    )

    # Buy markers
    buy_trades = [t for t in trades if 'Exit Date' not in t or t['Type'] == 'BUY']
    if buy_trades:
        buy_dates = [t['Entry Date'] for t in buy_trades]
        buy_prices = [t['Entry Price'] for t in buy_trades]
        buy_values = [initial_capital * 0.95 for _ in buy_dates]  # near bottom
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_values,
                mode='markers',
                marker=dict(size=10, color='green', symbol='triangle-up'),
                name='Buy Signal',
            ),
            row=1, col=1,
        )

    # Sell markers
    sell_trades = [t for t in trades if 'Exit Date' in t and t['Type'] in ('SELL', 'STRONG_SELL')]
    if sell_trades:
        sell_dates = [t['Exit Date'] for t in sell_trades]
        sell_values = [max(portfolio_values) * 0.95 for _ in sell_dates]  # near top
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_values,
                mode='markers',
                marker=dict(size=10, color='red', symbol='triangle-down'),
                name='Sell Signal',
            ),
            row=1, col=1,
        )

    # Price reference (bottom panel)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            name=f'{ticker} Price',
            line=dict(color='#FFA500', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 165, 0, 0.1)',
        ),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="Portfolio Value (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Stock Price (₹)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.update_layout(
        title=f'{ticker} Backtest Equity Curve',
        hovermode='x unified',
        height=600,
        template='plotly_dark',
    )

    return fig


# Needed for make_subplots
from plotly.subplots import make_subplots
