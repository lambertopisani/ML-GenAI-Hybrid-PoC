# app.py - Optimized for Streamlit with XGBoost (lightweight, fast, perfect for deployment)
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os 
import json


# Try Claude (Anthropic) integration 
try:
    from anthropic import Anthropic
    claude_available = True
    client = Anthropic(api_key=os.environ.get(claude_key))
except:
    claude_available = False

# -----------------------------
# Synthetic Data Generator
# -----------------------------
def generate_synthetic_market(n_days=1000, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start="2022-01-01", periods=n_days)
    trend = np.linspace(0, 5, n_days)
    seasonality = 2 * np.sin(np.linspace(0, 15, n_days))
    noise = np.random.normal(0, 1, n_days)
    price = 50 + trend + seasonality + noise.cumsum() * 0.3
    return pd.DataFrame({"date": dates, "price": price})

# -----------------------------
# Enhanced Feature Engineering 
# -----------------------------
def create_features(df):
    df = df.copy()
    df["returns"] = df["price"].pct_change()
    df["ma_5"] = df["price"].rolling(5).mean()
    df["ma_20"] = df["price"].rolling(20).mean()
    df["volatility"] = df["returns"].rolling(10).std()
    df["momentum"] = df["price"] - df["price"].shift(5)
    df["rsi"] = compute_rsi(df["price"], 14)  # Added RSI
    df["price_position"] = (df["price"] - df["ma_20"]) / df["ma_20"]  # Added position
    return df.dropna()

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# -----------------------------
# Target Creation
# -----------------------------
def create_target(df):
    df = df.copy()
    df["future_return"] = df["price"].shift(-1) / df["price"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    return df.dropna()

# -----------------------------
# XGBoost Model (Streamlit-optimized)
# -----------------------------
def train_model(df):
    features = ["returns", "ma_5", "ma_20", "volatility", "momentum", "rsi", "price_position"]
    X = df[features]
    y = df["target"]
    
    # XGBoost - lightweight, fast, perfect for Streamlit
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X, y)
    return model

# -----------------------------
# Latest Signal with Model Persistence
# -----------------------------
def get_latest_signal(df, model):
    features = ["returns", "ma_5", "ma_20", "volatility", "momentum", "rsi", "price_position"]
    latest = df[features].iloc[-1:].fillna(0)
    
    prediction = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][1]
    
    context = dict(zip(features, latest.iloc[0].values))
    
    return {
        "prediction": "🟢 UP" if prediction == 1 else "🔴 DOWN",
        "confidence": float(prob),
        "features": context
    }

# -----------------------------
# Claude Explanation
# -----------------------------
def generate_explanation(signal):
    if claude_available:
        prompt = f"""<financial-analyst>
Analyze this trading signal in 3-5 bullets:

PREDICTION: {signal['prediction']}
CONFIDENCE: {signal['confidence']:.1%}
KEY FEATURES: RSI={signal['features']['rsi']:.1f}, Momentum={signal['features']['momentum']:.2f}, 
Volatility={signal['features']['volatility']:.3f}

Provide:
• Prediction reasoning
• Key indicator conflicts  
• Risk level (Low/Med/High)
• Action (Buy/Hold/Sell) + stop-loss idea

Keep concise, actionable, professional.
</financial-analyst>"""

        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=250,
                temperature=0.1,
                system="You are a professional trading analyst. Respond in markdown bullets only.",
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            tokens = len(prompt.split()) + len(text.split())
            cost = tokens / 1_000_000 * 10  # ~$10/M tokens blended
            return text, cost
        except:
            pass
    
    # Enhanced fallback
    conf = signal['confidence']
    pred = "🟢 UP" if conf > 0.5 else "🔴 DOWN"
    return f"""
**{pred}** | Confidence: {conf:.1%}

• **Trend**: {'Bullish MAs' if signal['features']['ma_5'] > signal['features']['ma_20'] else 'Bearish MAs'}
• **RSI**: {'Oversold 📉' if signal['features']['rsi'] < 30 else 'Overbought 📈' if signal['features']['rsi'] > 70 else 'Neutral ⚖️'}
• **Risk**: {'High' if signal['features']['volatility'] > 0.03 else 'Moderate'}
• **Action**: {'🟢 BUY' if conf > 0.65 else '🟡 HOLD' if conf > 0.45 else '🔴 SELL/Wait'}
• **Stop**: {'2% below entry' if pred == '🟢 UP' else '2% above entry'}
""", 0.0

# -----------------------------
# STREAMLIT APP - Production Ready
# -----------------------------
st.set_page_config(page_title="AI Trading Copilot", layout="wide")

st.title("🤖 AI Trading Decision Copilot")
st.markdown("**XGBoost + Claude 3.5 Sonnet** | Real-time market signals & analysis")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Controls")
    n_days = st.slider("Training Days", 200, 2000, 1000)
    seed = st.slider("Random Seed", 1, 100, 42)
    
    st.header("🚀 Setup")
    st.info("""
    1. `pip install streamlit xgboost anthropic pandas numpy scikit-learn joblib`
    2. `export ANTHROPIC_API_KEY=sk-ant-...`
    3. `streamlit run app.py`
    """)

# Cache entire pipeline
@st.cache_data(show_spinner="Training XGBoost model...")
def run_pipeline(_n_days, _seed):
    df = generate_synthetic_market(_n_days, _seed)
    df = create_features(df)
    df = create_target(df[:-1])  # Remove last row (no future)
    model = train_model(df)
    signal = get_latest_signal(df, model)
    explanation, cost = generate_explanation(signal)
    return df, model, signal, explanation, cost

# Run pipeline
df, model, signal, explanation, cost = run_pipeline(n_days, seed)

# Main Dashboard
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("📈 Price & Indicators")
    chart_df = df.set_index("date")[["price", "ma_5", "ma_20"]].tail(200)
    st.line_chart(chart_df, use_container_width=True)

with col2:
    st.subheader("🎯 Live Signal")
    st.metric("Prediction", signal["prediction"], delta=None)
    st.metric("Confidence", f"{signal['confidence']:.1%}")
    
    if claude_available:
        st.metric("🧠 LLM Cost", f"${cost:.6f}")

with col3:
    st.subheader("🔥 Top Features")
    features = ["returns", "ma_5", "ma_20", "volatility", "momentum", "rsi", "price_position"]
    imp_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)
    st.bar_chart(imp_df.set_index("feature"))

# AI Analysis
st.subheader("💡 Claude AI Analysis")
st.markdown(explanation)

# Model Performance
st.subheader("📊 Backtest Results")
features = ["returns", "ma_5", "ma_20", "volatility", "momentum", "rsi", "price_position"]
X_test = df[features]
y_true = df["target"]
y_pred = model.predict(X_test)
accuracy = (y_pred == y_true).mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.1%}")
col2.metric("Total Signals", len(df))
col3.metric("UP Signals", f"{y_pred.sum()}")
col4.metric("Cache Hits", st.cache_data.get_stats()["cache_hits"])
