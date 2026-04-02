# app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

# Try OpenAI integration
try:
    from openai import OpenAI
    openai_available = True
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except:
    openai_available = False

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
# Feature Engineering
# -----------------------------
def create_features(df):
    df["returns"] = df["price"].pct_change()
    df["ma_5"] = df["price"].rolling(5).mean()
    df["ma_20"] = df["price"].rolling(20).mean()
    df["volatility"] = df["returns"].rolling(10).std()
    df["momentum"] = df["price"] - df["price"].shift(5)
    return df.dropna()

# -----------------------------
# Target
# -----------------------------
def create_target(df):
    df["future_return"] = df["price"].shift(-1) / df["price"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    return df.dropna()

# -----------------------------
# Train Model
# -----------------------------
def train_model(df):
    features = ["returns","ma_5","ma_20","volatility","momentum"]
    X = df[features]
    y = df["target"]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

# -----------------------------
# Get Latest Signal
# -----------------------------
def get_latest_signal(df, model):
    features = ["returns","ma_5","ma_20","volatility","momentum"]
    latest = df.iloc[-1:]
    prediction = model.predict(latest[features])[0]
    prob = model.predict_proba(latest[features])[0][1]
    context = latest[features].to_dict(orient="records")[0]
    return {"prediction":"UP" if prediction==1 else "DOWN","confidence":float(prob),"features":context}

# -----------------------------
# LLM Explanation
# -----------------------------
def generate_explanation(signal):
    # If OpenAI available, use real GPT
    if openai_available:
        prompt = f"""
You are a financial decision assistant.
Prediction: {signal['prediction']}
Confidence: {signal['confidence']:.2f}
Features: {signal['features']}
Tasks:
1. Explain prediction reasoning
2. Highlight conflicting indicators
3. Risk assessment (Low/Med/High)
4. Suggested action (Buy/Hold/Sell)
5. Keep concise 3-5 bullets
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.3
            )
            text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, "usage") else None
            cost_estimate = tokens_used/1000*0.03 if tokens_used else None
            return text, cost_estimate
        except:
            pass
    # Fallback mock explanation
    text = f"""
**Prediction:** {signal['prediction']}  
**Confidence:** {round(signal['confidence'],2)}  
- Momentum is {'positive' if signal['features']['momentum']>0 else 'negative'}  
- Short vs long MA trend: {'bullish' if signal['features']['ma_5']>signal['features']['ma_20'] else 'bearish'}  
- Volatility: {'high' if signal['features']['volatility']>0.03 else 'moderate'}  
- Risk: {'High' if signal['features']['volatility']>0.03 else 'Moderate'}  
- Action: {'Buy' if signal['prediction']=='UP' and signal['confidence']>0.6 else 'Hold/Wait'}
"""
    return text, 0.0

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📈 AI Trading Decision Copilot (Hybrid ML + GenAI)")

# Inputs
n_days = st.slider("Number of Days", 200, 1000, 1000)
seed = st.slider("Random Seed", 1, 100, 42)

# Generate data
df = generate_synthetic_market(n_days, seed)
df = create_features(df)
df = create_target(df)
model = train_model(df)
signal = get_latest_signal(df, model)
explanation, cost = generate_explanation(signal)

# Display
st.subheader("Market Price")
st.line_chart(df.set_index("date")["price"])

st.subheader("Model Signal")
st.json(signal)

st.subheader("AI Explanation")
st.markdown(explanation)

st.subheader("Estimated LLM Cost (USD)")
st.write(f"${cost:.4f}" if cost else "N/A")
