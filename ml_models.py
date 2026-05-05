# ml_models.py
# XGBoost-based price direction prediction model

import os
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def compute_features_from_hist(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 20+ technical features from OHLCV history.
    Returns DataFrame with indicator-based features (NaN rows dropped).
    """
    if len(hist) < 50:
        return pd.DataFrame()

    close = hist['Close']
    volume = hist['Volume'] if 'Volume' in hist.columns else pd.Series([1] * len(hist))

    # SMAs
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
    bb_pct = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    bb_width = (bb_upper - bb_lower) / bb_middle

    # Price distance from SMAs
    price_vs_sma20 = (close / sma20 - 1) * 100
    price_vs_sma50 = (close / sma50 - 1) * 100
    price_vs_sma200 = (close / sma200 - 1) * 100

    # Volume
    volume_ma20 = volume.rolling(20, min_periods=1).mean()
    volume_ratio = volume / volume_ma20.replace(0, 1)

    # Volatility
    returns = close.pct_change()
    volatility_20d = returns.rolling(20, min_periods=1).std() * np.sqrt(252)

    # Returns
    returns_5d = close.pct_change(5) * 100
    returns_10d = close.pct_change(10) * 100
    returns_20d = close.pct_change(20) * 100

    # 52-week highs/lows
    high_52w = close.rolling(252, min_periods=1).max()
    low_52w = close.rolling(252, min_periods=1).min()
    high_52w_pct = (close / high_52w - 1) * 100
    low_52w_pct = (close / low_52w - 1) * 100

    # Build feature DataFrame
    features_df = pd.DataFrame({
        'rsi_14': rsi,
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'macd_hist': macd_hist,
        'bb_pct': bb_pct,
        'bb_width': bb_width,
        'sma_20': sma20,
        'sma_50': sma50,
        'sma_200': sma200,
        'price_vs_sma20': price_vs_sma20,
        'price_vs_sma50': price_vs_sma50,
        'price_vs_sma200': price_vs_sma200,
        'volume_ratio': volume_ratio,
        'volatility_20d': volatility_20d,
        'returns_5d': returns_5d,
        'returns_10d': returns_10d,
        'returns_20d': returns_20d,
        'high_52w_pct': high_52w_pct,
        'low_52w_pct': low_52w_pct,
    }, index=hist.index)

    # Normalize SMAs to relative values
    features_df['sma_20'] = price_vs_sma20
    features_df['sma_50'] = price_vs_sma50
    features_df['sma_200'] = price_vs_sma200

    # Drop NaN rows
    features_df = features_df.dropna()

    return features_df


def build_training_dataset(tickers: list, horizon: int = 30, fetch_fn=None) -> tuple:
    """
    Build X (features) and y (labels) for training.
    Label: 1 if price N days later > current price, else 0.
    """
    if fetch_fn is None:
        return pd.DataFrame(), pd.Series()

    X_list = []
    y_list = []
    ticker_list = []

    for ticker in tickers:
        try:
            hist = fetch_fn(ticker)
            if hist is None or len(hist) < horizon + 50:
                continue

            features = compute_features_from_hist(hist)
            if features.empty or len(features) < horizon + 10:
                continue

            close = hist['Close']

            for i in range(len(features) - horizon):
                feature_row = features.iloc[i]
                future_idx = i + horizon
                if future_idx < len(close):
                    label = 1 if close.iloc[future_idx] > close.iloc[i] else 0
                    X_list.append(feature_row)
                    y_list.append(label)
                    ticker_list.append(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    if not X_list:
        return pd.DataFrame(), pd.Series()

    X = pd.DataFrame(X_list)
    y = pd.Series(y_list, name='target')

    return X, y


def train_model(tickers: list, horizon: int = 30, fetch_fn=None) -> dict:
    """
    Train XGBoost model or load from cache if < 24h old.
    Returns dict with model, scaler, feature_names, cv_accuracy, etc.
    """
    model_file = os.path.join(MODEL_DIR, f"xgb_cross_{horizon}d.joblib")

    # Check cache
    if os.path.exists(model_file):
        model_info = joblib.load(model_file)
        trained_at = model_info.get('trained_at')
        if trained_at:
            elapsed = datetime.now() - datetime.fromisoformat(trained_at)
            if elapsed < timedelta(hours=24):
                return model_info

    # Build dataset
    X, y = build_training_dataset(tickers, horizon, fetch_fn)

    if X.empty:
        return {
            'model': None,
            'scaler': None,
            'feature_names': [],
            'cv_accuracy': 0,
            'horizon': horizon,
            'trained_at': datetime.now().isoformat(),
            'error': 'Could not build training dataset',
        }

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        cv_scores.append(score)

    # Train final model on all data
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )
    model.fit(X_scaled, y)

    # Save to cache
    model_info = {
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'cv_accuracy': np.mean(cv_scores),
        'horizon': horizon,
        'trained_at': datetime.now().isoformat(),
        'num_samples': len(X),
    }

    joblib.dump(model_info, model_file)

    return model_info


def predict_stock(ticker: str, hist: pd.DataFrame, model_bundle: dict) -> dict:
    """
    Predict if stock will be up in N days based on latest features.
    """
    if model_bundle['model'] is None:
        return {
            'ticker': ticker,
            'probability_up': 0.5,
            'direction': 'UNKNOWN',
            'confidence_label': 'Low',
            'horizon_days': model_bundle['horizon'],
            'top_features': [],
            'error': 'Model not trained',
        }

    features = compute_features_from_hist(hist)
    if features.empty:
        return {
            'ticker': ticker,
            'probability_up': 0.5,
            'direction': 'UNKNOWN',
            'confidence_label': 'Low',
            'horizon_days': model_bundle['horizon'],
            'top_features': [],
            'error': 'Could not compute features',
        }

    # Last row of features
    last_features = features.iloc[-1:][model_bundle['feature_names']]
    last_features_scaled = model_bundle['scaler'].transform(last_features)

    # Predict
    prob = model_bundle['model'].predict_proba(last_features_scaled)[0][1]
    direction = 'UP' if prob > 0.5 else 'DOWN'

    # Confidence
    if prob > 0.65 or prob < 0.35:
        confidence_label = 'High'
    elif 0.45 <= prob <= 0.65 or 0.35 <= prob < 0.45:
        confidence_label = 'Medium'
    else:
        confidence_label = 'Low'

    # Feature importance
    importances = model_bundle['model'].feature_importances_
    feature_importance_dict = dict(zip(model_bundle['feature_names'], importances))
    top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        'ticker': ticker,
        'probability_up': prob,
        'direction': direction,
        'confidence_label': confidence_label,
        'horizon_days': model_bundle['horizon'],
        'top_features': top_features,
    }


def get_feature_importance_chart(model_bundle: dict) -> go.Figure:
    """
    Bar chart of top 15 feature importances.
    """
    if model_bundle['model'] is None:
        return go.Figure()

    importances = model_bundle['model'].feature_importances_
    feature_names = model_bundle['feature_names']

    # Top 15
    importance_dict = dict(zip(feature_names, importances))
    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]

    fig = go.Figure(
        data=go.Bar(
            x=values,
            y=names,
            orientation='h',
            marker=dict(color='#00D9FF'),
        )
    )

    fig.update_layout(
        title='XGBoost Feature Importances (Top 15)',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=500,
        template='plotly_dark',
        showlegend=False,
    )

    return fig
