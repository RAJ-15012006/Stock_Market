import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from financial_agent import agent_team, run_debate, run_morning_brief, get_morning_briefs, PortfolioTool, set_model, run_agent_with_retry, MODEL as DEFAULT_MODEL
from agno.models.groq import Groq
import threading
import json
import os
from ml_engine import train_prediction_model, get_shap_explanation, detect_market_regimes, run_strategy_backtest, RLOptimizerAgent, finbert_analyzer
from rag_engine import rag_assistant

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Agentic AI: 3D Market Intelligence",
    page_icon="🤖",
    layout="wide",
)

# --- CUSTOM CSS (ShadCN-inspired) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background-color: #09090b; }
    .stApp { background-color: #09090b; color: #fafafa; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; color: #00f2ff; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem; }
    .stChatFloatingInputContainer { background-color: rgba(9,9,11,0.9); backdrop-filter: blur(12px); border-top: 1px solid #27272a; }
    .stChatMessage { background: #18181b; border: 1px solid #27272a; border-radius: 12px; }
    .reasoning-step {
        background: rgba(0, 242, 255, 0.07);
        border-left: 3px solid #00f2ff;
        padding: 10px 14px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 0.8rem;
        border-radius: 0 8px 8px 0;
    }
    .card {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        transition: border-color 0.2s;
    }
    .card:hover { border-color: #00f2ff44; }
    .card-title { font-size: 0.75rem; font-weight: 600; color: #71717a; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
    .card-value { font-size: 1.8rem; font-weight: 700; color: #fafafa; }
    .card-sub { font-size: 0.85rem; color: #71717a; margin-top: 4px; }
    .badge-green { background: #14532d; color: #4ade80; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .badge-red { background: #450a0a; color: #f87171; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .badge-blue { background: #0c1a3a; color: #60a5fa; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .hero-title { font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #00f2ff, #a855f7, #f97316); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2; }
    .hero-sub { font-size: 1rem; color: #71717a; margin-top: 8px; }
    .ticker-bar { background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 10px 16px; font-family: monospace; font-size: 0.85rem; overflow: hidden; white-space: nowrap; }
    .agent-card {
        background: linear-gradient(135deg, #18181b, #1c1c1f);
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .agent-icon { font-size: 2rem; margin-bottom: 8px; }
    .agent-name { font-size: 0.85rem; font-weight: 600; color: #fafafa; }
    .agent-role { font-size: 0.75rem; color: #71717a; margin-top: 2px; }
    .divider { border: none; border-top: 1px solid #27272a; margin: 20px 0; }
    section[data-testid="stSidebar"] { background: #09090b; border-right: 1px solid #27272a; }
    </style>
""", unsafe_allow_html=True)

# --- 3D GLOBE COMPONENT (Three.js) ---
def render_3d_globe():
    three_js_code = """
    <div id="container" style="width: 100%; height: 400px; background: #0e1117;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 400, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, 400);
        document.getElementById('container').appendChild(renderer.domElement);

        const geometry = new THREE.SphereGeometry(2, 32, 32);
        const material = new THREE.MeshPhongMaterial({
            color: 0x00f2ff,
            wireframe: true,
            transparent: true,
            opacity: 0.5
        });
        const globe = new THREE.Mesh(geometry, material);
        scene.add(globe);

        const light = new THREE.PointLight(0xffffff, 1, 100);
        light.position.set(10, 10, 10);
        scene.add(light);
        scene.add(new THREE.AmbientLight(0x404040));

        camera.position.z = 5;

        function animate() {
            requestAnimationFrame(animate);
            globe.rotation.y += 0.005;
            globe.rotation.x += 0.002;
            renderer.render(scene, camera);
        }
        animate();
    </script>
    """
    st.components.v1.html(three_js_code, height=400)

# --- MAIN UI ---

# Hero Section
st.markdown("""
<div style="padding: 32px 0 16px 0;">
    <div class="hero-title">Agentic 3D Financial<br>Intelligence Collective</div>
    <div class="hero-sub">Powered by Multi-Agent AI · Real-time Market Data · Voice Ready</div>
</div>
""", unsafe_allow_html=True)

# Live Ticker Bar
st.markdown("""
<div class="ticker-bar">
    <marquee behavior="scroll" direction="left" scrollamount="4">
        &nbsp;&nbsp;
        🟢 NVDA $875.40 +2.1%&nbsp;&nbsp;|&nbsp;&nbsp;
        🔴 AAPL $189.30 -0.5%&nbsp;&nbsp;|&nbsp;&nbsp;
        🟢 BTC $63,200 +1.2%&nbsp;&nbsp;|&nbsp;&nbsp;
        🟢 TSLA $177.80 +0.8%&nbsp;&nbsp;|&nbsp;&nbsp;
        🔴 META $485.20 -0.3%&nbsp;&nbsp;|&nbsp;&nbsp;
        🟢 MSFT $415.60 +1.1%&nbsp;&nbsp;|&nbsp;&nbsp;
        🟢 GOOGL $172.40 +0.9%&nbsp;&nbsp;|&nbsp;&nbsp;
        🟢 ETH $3,100 +0.5%&nbsp;&nbsp;|&nbsp;&nbsp;
        🇮🇳 🟢 RELIANCE ₹2,945 +1.3%&nbsp;&nbsp;|&nbsp;&nbsp;
        🇮🇳 🟢 TCS ₹3,812 +0.9%&nbsp;&nbsp;|&nbsp;&nbsp;
        🇮🇳 🔴 INFY ₹1,478 -0.4%&nbsp;&nbsp;|&nbsp;&nbsp;
        🇮🇳 🟢 HDFCBANK ₹1,623 +0.7%&nbsp;&nbsp;|&nbsp;&nbsp;
        🇮🇳 🟢 WIPRO ₹462 +1.1%&nbsp;&nbsp;|&nbsp;&nbsp;
        🇮🇳 🔴 BAJFINANCE ₹6,890 -0.6%&nbsp;&nbsp;|&nbsp;&nbsp;
        🇮🇳 🟢 TATAMOTORS ₹978 +2.3%&nbsp;&nbsp;|&nbsp;&nbsp;
        🇮🇳 🟢 SBIN ₹812 +0.5%&nbsp;&nbsp;
    </marquee>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin: 16px 0'></div>", unsafe_allow_html=True)

# Top Stats Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">🤖 Active Agents</div>
        <div class="card-value">3</div>
        <div class="card-sub">Finance · Web · Sentiment</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">💬 Session Queries</div>
        <div class="card-value" id="qcount">{}</div>
        <div class="card-sub">This session</div>
    </div>""".format(len(st.session_state.get('messages', []))//2), unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">📡 Data Sources</div>
        <div class="card-value">2</div>
        <div class="card-sub">YFinance · DuckDuckGo</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">🧠 LLM Model</div>
        <div class="card-value" style="font-size:1.1rem;">{st.session_state.get('current_model_name', 'Llama 3.1')}</div>
        <div class="card-sub">{st.session_state.get('current_model_tier', '8B · Instant')}</div>
    </div>""", unsafe_allow_html=True)


# Globe + Agent Cards Row
globe_col, agents_col = st.columns([2, 1])

with globe_col:
    render_3d_globe()

with agents_col:
    st.markdown("""
    <div class="card" style="margin-bottom:12px;">
        <div class="card-title">⚡ Agent Collective Status</div>
        <div style="display:flex; flex-direction:column; gap:10px; margin-top:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.9rem;">📈 Finance Agent</span>
                <span class="badge-green">● ACTIVE</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.9rem;">🔍 Web Search Agent</span>
                <span class="badge-green">● ACTIVE</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.9rem;">🧠 Sentiment Agent</span>
                <span class="badge-green">● ACTIVE</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.9rem;">🗄️ SQLite Memory</span>
                <span class="badge-blue">● READY</span>
            </div>
        </div>
    </div>
    <div class="card">
        <div class="card-title">💼 Portfolio Snapshot</div>
        <div style="display:flex; flex-direction:column; gap:8px; margin-top:10px;">
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:0.85rem; color:#a1a1aa;">NVDA × 10</span>
                <span class="badge-green">+2.1%</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:0.85rem; color:#a1a1aa;">AAPL × 5</span>
                <span class="badge-red">-0.5%</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:0.85rem; color:#a1a1aa;">BTC × 0.5</span>
                <span class="badge-green">+1.2%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Quick Prompt Suggestions
st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#71717a; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>💡 Try asking...</div>", unsafe_allow_html=True)
sugg_cols = st.columns(4)
suggestions = [
    "Analyze NVDA stock",
    "Analyze RELIANCE.NS stock",
    "Compare TCS.NS vs INFY.NS",
    "Market sentiment today",
    "Latest crypto news",
    "Compare AAPL vs MSFT",
    "Top Indian IT stocks",
    "Nifty 50 outlook",
]
for i, (col, s) in enumerate(zip(sugg_cols, suggestions)):
    with col:
        if st.button(s, key=f"sugg_{i}"):
            st.session_state.suggested_prompt = s

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_stock_data(symbol, period):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    info = {}
    try:
        fi = stock.fast_info
        info = {k: getattr(fi, k, None) for k in ['last_price', 'market_cap', 'year_high', 'year_low']}
    except Exception:
        pass
    return hist, info

US_SYMBOLS    = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "BTC-USD", "ETH-USD", "NFLX"]
INDIA_SYMBOLS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS",
                 "ICICIBANK.NS", "BAJFINANCE.NS", "ADANIENT.NS", "SBIN.NS",
                 "TATAMOTORS.NS", "HINDUNILVR.NS", "MARUTI.NS", "SUNPHARMA.NS",
                 "ONGC.NS", "AXISBANK.NS"]

def get_currency(sym): return "\u20b9" if sym.endswith((".NS", ".BO")) else "$"

# Sidebar: Market Dashboard
with st.sidebar:
    _banner = os.path.join(os.path.dirname(__file__), "assets", "banner.png")
    if os.path.exists(_banner):
        st.image(_banner, width='stretch')
    st.header("📈 Market Pulse")
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    
    model_options = {
        "Llama 3.3 70B (Powerful & Versatile)": "llama-3.3-70b-versatile",
        "Gemini 1.5 Flash (Backup)": "gemini-1.5-flash",
        "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
        "Qwen 2.5 Coder 7B (Local)": "qwen2.5-coder:7b",
        "DeepSeek R1 1.5B (Local)": "deepseek-r1:1.5b",
    }
    
    selected_model_name = st.selectbox("Select Agent Model", list(model_options.keys()), index=0)
    selected_model_id = model_options[selected_model_name]
    
    if "current_model_id" not in st.session_state or st.session_state.current_model_id != selected_model_id:
        set_model(selected_model_id)
        st.session_state.current_model_id = selected_model_id
        st.session_state.current_model_name = selected_model_name.split(" (")[0]
        st.session_state.current_model_tier = "70B · Versatile" if "70B" in selected_model_name else "Gemini · Flash" if "Gemini" in selected_model_name else "8B · Instant" if "8B" in selected_model_name else "SMoE · 32k"
        st.toast(f"Switched to {selected_model_name}")

    st.markdown("---")
    st.subheader("🔍 Symbol Selection")
    market = st.radio("Market", ["🌍 Global (US/Crypto)", "🇮🇳 India (NSE)"], horizontal=True, key="sb_market")
    DEFAULT_SYMBOLS = INDIA_SYMBOLS if "India" in market else US_SYMBOLS
    symbol = st.selectbox("Quick Select", DEFAULT_SYMBOLS)
    custom = st.text_input("Or type custom symbol", "", help="US: AAPL | India: RELIANCE.NS | BSE: RELIANCE.BO").upper().strip()
    if custom:
        if "India" in market and not custom.endswith((".NS", ".BO", "-USD")):
            custom += ".NS"
        symbol = custom

    period = st.radio("Period", ["1wk", "1mo", "3mo", "6mo", "1y"], index=1, horizontal=True)
    compare = st.text_input("Compare with (e.g. AAPL)", "").upper().strip()

    if symbol:
        try:
            hist, info = get_stock_data(symbol, period)
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else price
                change = ((price - prev) / prev) * 100
                cur = get_currency(symbol)
                st.metric("Price", f"{cur}{price:.2f}", f"{change:+.2f}%")

                col1, col2 = st.columns(2)
                col1.metric("High", f"{cur}{hist['High'].max():.2f}")
                col2.metric("Low",  f"{cur}{hist['Low'].min():.2f}")
                col1.metric("Volume", f"{int(hist['Volume'].iloc[-1]):,}")
                col2.metric("Avg Vol", f"{int(hist['Volume'].mean()):,}")

                try:
                    mc = info.get('market_cap')
                    yh = info.get('year_high')
                    col1.metric("Market Cap", f"{cur}{mc/1e9:.1f}B" if mc else "N/A")
                    col2.metric("52W High",   f"{cur}{yh:.2f}" if yh else "N/A")
                except Exception:
                    pass

                # Candlestick chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name=symbol))

                # MA20 & MA50
                if len(hist) >= 20:
                    hist['MA20'] = hist['Close'].rolling(20).mean()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='MA20', line=dict(color='orange', width=1)))
                if len(hist) >= 50:
                    hist['MA50'] = hist['Close'].rolling(50).mean()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], name='MA50', line=dict(color='cyan', width=1)))

                # Compare symbol
                if compare:
                    try:
                        hist2, _ = get_stock_data(compare, period)
                        if not hist2.empty:
                            fig.add_trace(go.Scatter(x=hist2.index, y=hist2['Close'], name=compare, line=dict(color='lime', width=1)))
                    except Exception:
                        pass

                fig.update_layout(template="plotly_dark", height=280, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, width='stretch')

                # RSI
                if len(hist) >= 14:
                    delta = hist['Close'].diff()
                    gain = delta.clip(lower=0).rolling(14).mean()
                    loss = -delta.clip(upper=0).rolling(14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi_val = rsi.iloc[-1]
                    rsi_color = "🟢" if rsi_val < 30 else "🔴" if rsi_val > 70 else "🟡"
                    st.metric(f"{rsi_color} RSI (14)", f"{rsi_val:.1f}")

                # News
                st.markdown("**📰 Latest News**")
                try:
                    news = yf.Ticker(symbol).news
                    for n in (news or [])[:3]:
                        title = n.get('content', {}).get('title', '') or n.get('title', '')
                        url = n.get('content', {}).get('canonicalUrl', {}).get('url', '') or n.get('link', '')
                        if title and url:
                            st.markdown(f"- [{title[:50]}...]({url})")
                except Exception:
                    st.caption("News unavailable")

                # Analyst recommendations
                st.markdown("**🎯 Analyst Recommendations**")
                try:
                    rec = yf.Ticker(symbol).recommendations
                    if rec is not None and not rec.empty:
                        latest = rec.iloc[-1]
                        st.write(latest.to_frame().T)
                    else:
                        st.caption("No recommendations available")
                except Exception:
                    st.caption("Recommendations unavailable")
            else:
                st.warning(f"No data for {symbol}. Try adding -USD for crypto (e.g. BTC-USD)")
        except Exception as e:
            st.warning(f"Error: {e}")

    st.markdown("---")
    st.subheader("💼 Simulated Portfolio")
    st.write("NVDA: 10 | AAPL: 5 | BTC: 0.5")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab_chat, tab_candle, tab_backtest, tab_market, tab_portfolio, tab_screener, tab_debate, tab_trading, tab_schedule, tab_rag, tab_earnings, tab_rl = st.tabs([
    "🤖 AI Chat", "🕯️ Candle Trading", "📊 Backtest & Regimes", "🌐 Market Data",
    "💼 Portfolio", "🔍 AI Screener", "⚔️ Agent Debate", "📊 Auto Trading", "⏰ Scheduled Briefs",
    "🧠 RAG Research", "🎙️ Earnings Analyzer", "🤖 RL Optimization"
])

# ── TAB: CANDLE TRADING ──────────────────────────────────────────────────────
with tab_candle:
    st.markdown("### 📈 Advanced Trading Studio")
    st.caption("8 chart types · RSI · MACD · Bollinger Bands · Buy/Sell signals · One-click trade")

    cc1, cc2, cc3, cc4, cc5 = st.columns([2, 1, 1, 1, 1])
    with cc1:
        ct_symbol = st.text_input("Symbol", value="NVDA", key="ct_sym").upper().strip()
    with cc2:
        ct_period = st.selectbox("Period", ["5d", "1mo", "3mo", "6mo", "1y"], index=1, key="ct_period")
    with cc3:
        ct_interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m"], index=0, key="ct_interval")
    with cc4:
        chart_type = st.selectbox("Chart Type", [
            "🕯️ Candlestick",
            "📊 OHLC Bar",
            "📉 Line",
            "🌄 Area",
            "🔵 Heikin-Ashi",
            "📦 Hollow Candle",
            "🎯 Dot/Scatter",
            "🏔️ Mountain",
        ], index=0, key="ct_chart_type")
    with cc5:
        show_signals = st.toggle("Show Signals", value=True, key="ct_signals")

    if ct_symbol:
        try:
            @st.cache_data(ttl=60)
            def load_candle_data(sym, per, ivl):
                return yf.Ticker(sym).history(period=per, interval=ivl)

            df = load_candle_data(ct_symbol, ct_period, ct_interval)

            if df.empty:
                st.warning(f"No data for {ct_symbol}")
            else:
                df = df.copy()
                # Indicators
                df["EMA9"]    = df["Close"].ewm(span=9).mean()
                df["MA20"]    = df["Close"].rolling(20).mean()
                df["MA50"]    = df["Close"].rolling(50).mean()
                delta         = df["Close"].diff()
                gain          = delta.clip(lower=0).rolling(14).mean()
                loss          = -delta.clip(upper=0).rolling(14).mean()
                df["RSI"]     = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
                ema12         = df["Close"].ewm(span=12).mean()
                ema26         = df["Close"].ewm(span=26).mean()
                df["MACD"]    = ema12 - ema26
                df["MACDsig"] = df["MACD"].ewm(span=9).mean()
                df["MACDhist"]= df["MACD"] - df["MACDsig"]
                df["BB_mid"]  = df["Close"].rolling(20).mean()
                bb_std        = df["Close"].rolling(20).std()
                df["BB_up"]   = df["BB_mid"] + 2 * bb_std
                df["BB_lo"]   = df["BB_mid"] - 2 * bb_std
                df["buy_sig"] = (df["RSI"] < 35) & (df["MACD"] > df["MACDsig"])
                df["sell_sig"]= (df["RSI"] > 65) & (df["MACD"] < df["MACDsig"])

                price_now  = float(df["Close"].iloc[-1])
                rsi_now    = float(df["RSI"].iloc[-1])
                macd_now   = float(df["MACD"].iloc[-1])
                sig_now    = float(df["MACDsig"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else price_now
                pct_chg    = (price_now - prev_close) / prev_close * 100

                # Metrics row
                m1, m2, m3, m4, m5 = st.columns(5)
                ct_cur = get_currency(ct_symbol)
                m1.metric("Price",    f"{ct_cur}{price_now:.2f}",  f"{pct_chg:+.2f}%")
                m2.metric("RSI (14)", f"{rsi_now:.1f}",
                          "🟢 Oversold" if rsi_now < 30 else "🔴 Overbought" if rsi_now > 70 else "🟡 Neutral")
                m3.metric("MACD",     f"{macd_now:.3f}",
                          "⬆️ Bullish" if macd_now > sig_now else "⬇️ Bearish")
                m4.metric("Volume",   f"{int(df['Volume'].iloc[-1]):,}")
                m5.metric("Avg Vol",  f"{int(df['Volume'].mean()):,}")

                # Signal banner
                if show_signals:
                    if df["buy_sig"].iloc[-1]:
                        st.success("🟢 BUY SIGNAL — RSI oversold + MACD bullish crossover")
                    elif df["sell_sig"].iloc[-1]:
                        st.error("🔴 SELL SIGNAL — RSI overbought + MACD bearish crossover")
                    else:
                        st.info("🟡 HOLD — No strong signal right now")

                # XGBoost ML Prediction & SHAP Explainability
                try:
                    with st.spinner("Calculating ML Forecast & SHAP values..."):
                        ml_hist = yf.Ticker(ct_symbol).history(period="1y")
                        ml_res, ml_err = train_prediction_model(ml_hist)
                    
                    if not ml_err and ml_res:
                        metrics = ml_res["metrics"]
                        prob = metrics["pred_prob"]
                        label = metrics["pred_label"]
                        
                        st.markdown("### 🧠 Machine Learning Forecast & Explainable AI (XGBoost + SHAP)")
                        
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        with mc1:
                            border_color = "00ff00" if label == "BUY/UP" else "ff0000"
                            text_color = "#4ade80" if label == "BUY/UP" else "#f87171"
                            st.markdown(f"""
                            <div class="card" style="border: 1px solid #{border_color}44;">
                                <div class="card-title">XGBoost Forecast</div>
                                <div class="card-value" style="color:{text_color};">{label}</div>
                                <div class="card-sub">Next 5 Trading Days</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with mc2:
                            st.markdown(f"""
                            <div class="card">
                                <div class="card-title">Buy Probability</div>
                                <div class="card-value">{prob*100:.1f}%</div>
                                <div class="card-sub">Model confidence score</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with mc3:
                            st.markdown(f"""
                            <div class="card">
                                <div class="card-title">Test Accuracy</div>
                                <div class="card-value">{metrics['accuracy']*100:.1f}%</div>
                                <div class="card-sub">Historical test accuracy</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with mc4:
                            st.markdown(f"""
                            <div class="card">
                                <div class="card-title">Model F1 Score</div>
                                <div class="card-value">{metrics['f1']*100:.1f}%</div>
                                <div class="card-sub">Harmonic mean of prec/rec</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Show SHAP Plot
                        shap_fig = get_shap_explanation(ml_res)
                        if shap_fig:
                            sh1, sh2 = st.columns([2, 1])
                            with sh1:
                                st.pyplot(shap_fig)
                            with sh2:
                                st.markdown("""
                                **SHAP Feature Interpretation:**
                                - **RSI/MACD**: Measures price momentum.
                                - **Volatility/Return**: Evaluates risk and recent returns.
                                - **Vol_Z**: Captures volume spikes indicating institutional interest.
                                - **Sentiment**: Integrates real-time FinBERT scores.
                                
                                *Positive values shift the forecast towards BUY, while negative values shift it towards SELL/HOLD.*
                                """)
                    else:
                        st.caption(f"ML forecasting offline for {ct_symbol}: {ml_err or 'No data'}")
                except Exception as e:
                    st.caption(f"Error computing ML prediction: {e}")

                # ── Main chart (type-switched) ────────────────────────────────────
                fig = go.Figure()

                if "Candlestick" in chart_type:
                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df["Open"], high=df["High"],
                        low=df["Low"], close=df["Close"], name=ct_symbol,
                        increasing_line_color="#4ade80", decreasing_line_color="#f87171",
                        increasing_fillcolor="#4ade80", decreasing_fillcolor="#f87171",
                    ))

                elif "OHLC" in chart_type:
                    fig.add_trace(go.Ohlc(
                        x=df.index, open=df["Open"], high=df["High"],
                        low=df["Low"], close=df["Close"], name=ct_symbol,
                        increasing_line_color="#4ade80", decreasing_line_color="#f87171",
                    ))

                elif "Line" in chart_type:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df["Close"], name=ct_symbol,
                        line=dict(color="#00f2ff", width=2),
                    ))

                elif "Area" in chart_type:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df["Close"], name=ct_symbol,
                        line=dict(color="#00f2ff", width=2),
                        fill="tozeroy", fillcolor="rgba(0,242,255,0.08)",
                    ))

                elif "Heikin-Ashi" in chart_type:
                    ha = df.copy()
                    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
                    ha["HA_Open"]  = (df["Open"].shift(1) + df["Close"].shift(1)) / 2
                    ha.at[ha.index[0], "HA_Open"] = df["Open"].iloc[0]
                    ha["HA_High"]  = ha[["HA_Open", "HA_Close"]].join(df["High"]).max(axis=1)
                    ha["HA_Low"]   = ha[["HA_Open", "HA_Close"]].join(df["Low"]).min(axis=1)
                    fig.add_trace(go.Candlestick(
                        x=ha.index, open=ha["HA_Open"], high=ha["HA_High"],
                        low=ha["HA_Low"], close=ha["HA_Close"], name="Heikin-Ashi",
                        increasing_line_color="#4ade80", decreasing_line_color="#f87171",
                        increasing_fillcolor="#4ade80", decreasing_fillcolor="#f87171",
                    ))

                elif "Hollow" in chart_type:
                    # Hollow candle: filled only when close < open
                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df["Open"], high=df["High"],
                        low=df["Low"], close=df["Close"], name=ct_symbol,
                        increasing_line_color="#4ade80", decreasing_line_color="#f87171",
                        increasing_fillcolor="rgba(0,0,0,0)", decreasing_fillcolor="#f87171",
                    ))

                elif "Dot" in chart_type or "Scatter" in chart_type:
                    colors = ["#4ade80" if c >= o else "#f87171"
                              for c, o in zip(df["Close"], df["Open"])]
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df["Close"], mode="markers", name=ct_symbol,
                        marker=dict(color=colors, size=6, symbol="circle"),
                    ))

                elif "Mountain" in chart_type:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df["Close"], name=ct_symbol,
                        line=dict(color="#a855f7", width=2),
                        fill="tozeroy", fillcolor="rgba(168,85,247,0.15)",
                    ))
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df["Low"], name="Low",
                        line=dict(color="rgba(168,85,247,0.3)", width=1),
                        fill="tonexty", fillcolor="rgba(168,85,247,0.05)",
                    ))

                # Overlays (MAs + BB) on all chart types
                fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA9",
                                         line=dict(color="#f97316", width=1.2)))
                fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20",
                                         line=dict(color="#a855f7", width=1.2)))
                fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50",
                                         line=dict(color="#00f2ff", width=1.2)))
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB Upper",
                                         line=dict(color="#71717a", width=1, dash="dot")))
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_lo"], name="BB Lower",
                                         line=dict(color="#71717a", width=1, dash="dot"),
                                         fill="tonexty", fillcolor="rgba(113,113,122,0.05)"))

                # Buy/Sell signal markers
                if show_signals:
                    buys  = df[df["buy_sig"]]
                    sells = df[df["sell_sig"]]
                    if not buys.empty:
                        fig.add_trace(go.Scatter(
                            x=buys.index, y=buys["Low"] * 0.99, mode="markers", name="BUY",
                            marker=dict(symbol="triangle-up", size=12, color="#4ade80")))
                    if not sells.empty:
                        fig.add_trace(go.Scatter(
                            x=sells.index, y=sells["High"] * 1.01, mode="markers", name="SELL",
                            marker=dict(symbol="triangle-down", size=12, color="#f87171")))

                fig.update_layout(
                    template="plotly_dark", height=480,
                    paper_bgcolor="#09090b", plot_bgcolor="#09090b",
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
                    xaxis=dict(gridcolor="#27272a"), yaxis=dict(gridcolor="#27272a"),
                )
                st.plotly_chart(fig, width='stretch')

                # ── Volume bars ───────────────────────────────────────────────
                vol_colors = ["#4ade80" if c >= o else "#f87171"
                              for c, o in zip(df["Close"], df["Open"])]
                fig_vol = go.Figure(go.Bar(
                    x=df.index, y=df["Volume"], marker_color=vol_colors, opacity=0.7))
                fig_vol.update_layout(
                    template="plotly_dark", height=110,
                    paper_bgcolor="#09090b", plot_bgcolor="#09090b",
                    margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                    xaxis=dict(gridcolor="#27272a"), yaxis=dict(gridcolor="#27272a"),
                )
                st.plotly_chart(fig_vol, width='stretch')

                # ── RSI + MACD ────────────────────────────────────────────────
                rsi_col, macd_col = st.columns(2)
                with rsi_col:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=df.index, y=df["RSI"], name="RSI",
                        line=dict(color="#a855f7", width=2),
                        fill="tozeroy", fillcolor="rgba(168,85,247,0.08)"))
                    fig_rsi.add_hline(y=70, line_dash="dot", line_color="#f87171", annotation_text="OB 70")
                    fig_rsi.add_hline(y=30, line_dash="dot", line_color="#4ade80", annotation_text="OS 30")
                    fig_rsi.add_hline(y=50, line_dash="dot", line_color="#71717a")
                    fig_rsi.update_layout(
                        template="plotly_dark", height=180, title="RSI (14)",
                        paper_bgcolor="#09090b", plot_bgcolor="#09090b",
                        margin=dict(l=0, r=0, t=30, b=0),
                        yaxis=dict(range=[0, 100], gridcolor="#27272a"),
                        xaxis=dict(gridcolor="#27272a"),
                    )
                    st.plotly_chart(fig_rsi, width='stretch')

                with macd_col:
                    hist_colors = ["#4ade80" if v >= 0 else "#f87171" for v in df["MACDhist"]]
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Bar(
                        x=df.index, y=df["MACDhist"],
                        marker_color=hist_colors, opacity=0.6, name="Histogram"))
                    fig_macd.add_trace(go.Scatter(
                        x=df.index, y=df["MACD"], name="MACD",
                        line=dict(color="#00f2ff", width=1.5)))
                    fig_macd.add_trace(go.Scatter(
                        x=df.index, y=df["MACDsig"], name="Signal",
                        line=dict(color="#f97316", width=1.5)))
                    fig_macd.update_layout(
                        template="plotly_dark", height=180, title="MACD (12/26/9)",
                        paper_bgcolor="#09090b", plot_bgcolor="#09090b",
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis=dict(gridcolor="#27272a"), yaxis=dict(gridcolor="#27272a"),
                    )
                    st.plotly_chart(fig_macd, width='stretch')

                # ── Trade Panel + P&L ─────────────────────────────────────────
                st.markdown("---")
                trade_col, pnl_col = st.columns(2)

                with trade_col:
                    st.markdown("**💸 Quick Trade**")
                    pt_c = PortfolioTool()
                    t1, t2, t3 = st.columns(3)
                    trade_qty    = t1.number_input("Qty", min_value=0.0001, value=1.0, step=0.5, key="ct_qty")
                    trade_action = t2.selectbox("Action", ["buy", "sell"], key="ct_action")
                    trade_reason = t3.text_input("Reason", value="Candle signal", key="ct_reason")
                    btn_label = f"⚡ {trade_action.upper()} {trade_qty} {ct_symbol}"
                    if st.button(btn_label, key="ct_execute", type="primary"):
                        res = pt_c.simulate_trade(ct_symbol, trade_qty, trade_action, trade_reason)
                        st.success(res) if trade_action == "buy" else st.error(res)
                        st.rerun()

                with pnl_col:
                    st.markdown("**📊 P&L Tracker**")
                    try:
                        holdings_pnl = json.loads(PortfolioTool().get_portfolio_balance())
                        if ct_symbol in holdings_pnl:
                            h = holdings_pnl[ct_symbol]
                            qty_held = h["quantity"]
                            cur_val  = h["value"]
                            all_logs = PortfolioTool.get_trade_log(100)
                            sym_logs = [l for l in all_logs if l["symbol"] == ct_symbol]
                            bought = sum(l["quantity"] for l in sym_logs if l["action"] == "buy")
                            sold   = sum(l["quantity"] for l in sym_logs if l["action"] == "sell")
                            cost   = bought * price_now - sold * price_now
                            pnl    = (cur_val - cost) if isinstance(cur_val, float) and cost else 0
                            pnl_pct= (pnl / cost * 100) if cost else 0
                            p1, p2, p3 = st.columns(3)
                            p1.metric("Held",  f"{qty_held:.4f}")
                            p2.metric("Value", f"{ct_cur}{cur_val:,.2f}" if isinstance(cur_val, float) else cur_val)
                            p3.metric("P&L",   f"${pnl:+,.2f}", f"{pnl_pct:+.1f}%")
                        else:
                            st.info(f"No {ct_symbol} position. Execute a BUY to start.")
                    except Exception as ex:
                        st.caption(f"P&L unavailable: {ex}")

                    st.markdown("**🗒️ Recent Trades**")
                    sym_trades = [l for l in PortfolioTool.get_trade_log(30) if l["symbol"] == ct_symbol]
                    if sym_trades:
                        for l in sym_trades[:6]:
                            badge = "badge-green" if l["action"] == "buy" else "badge-red"
                            st.markdown(
                                f"<div style='display:flex;justify-content:space-between;"
                                f"padding:4px 0;border-bottom:1px solid #27272a;font-size:0.8rem;'>"
                                f"<span style='color:#71717a;'>{l['ts'][:16]}</span>"
                                f"<span class='{badge}'>{l['action'].upper()}</span>"
                                f"<span>{l['quantity']:.4f}</span>"
                                f"<span style='color:#71717a;'>{l['reason'][:20]}</span>"
                                f"</div>", unsafe_allow_html=True)
                    else:
                        st.caption("No trades for this symbol yet.")

        except Exception as e:
            st.error(f"Chart error: {e}")

# ── TAB: BACKTEST & REGIMES ──────────────────────────────────────────────────
with tab_backtest:
    st.markdown("### 📊 Strategy Backtesting Arena & Market Regime HMM")
    st.caption("Test quantitative trading strategies and identify active market states using Hidden Markov Models (HMM).")

    bk_col1, bk_col2, bk_col3 = st.columns([2, 2, 1])
    with bk_col1:
        bk_symbol = st.text_input("Backtest Symbol", value="NVDA", key="bk_sym").upper().strip()
    with bk_col2:
        bk_strategy = st.selectbox("Trading Strategy", ["RSI", "MACD", "ML_Predictive"], key="bk_strategy")
    with bk_col3:
        bk_run = st.button("Run Backtest & HMM", key="bk_run_btn", type="primary")

    if bk_symbol:
        try:
            bk_hist = yf.Ticker(bk_symbol).history(period="1y")
            if bk_hist.empty:
                st.warning(f"No historical data found for {bk_symbol}")
            else:
                # 1. Run Strategy Backtest
                res_df, metrics = run_strategy_backtest(bk_hist, strategy=bk_strategy)
                
                st.markdown("#### 📈 Cumulative Returns vs Benchmark")
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=res_df.index, y=res_df['Cum_Strategy'], name=f"Strategy ({bk_strategy})", line=dict(color="#00f2ff", width=2)))
                fig_cum.add_trace(go.Scatter(x=res_df.index, y=res_df['Cum_Benchmark'], name="Benchmark (Buy & Hold)", line=dict(color="#71717a", width=1.5, dash="dot")))
                fig_cum.update_layout(template="plotly_dark", height=320, paper_bgcolor="#09090b", plot_bgcolor="#09090b", margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_cum, width='stretch')

                # Render metrics in columns
                bkc1, bkc2, bkc3, bkc4, bkc5 = st.columns(5)
                bkc1.metric("Strategy Total Return", f"{metrics['total_return_strat']:.1f}%")
                bkc2.metric("Benchmark Return", f"{metrics['total_return_bench']:.1f}%")
                bkc3.metric("Strategy Sharpe", f"{metrics['sharpe_strat']:.2f}")
                bkc4.metric("Strategy Max DD", f"{metrics['max_dd_strat']:.1f}%")
                bkc5.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

                st.markdown("---")

                # 2. Market Regime Detection using HMM
                st.markdown("#### ⏳ HMM Market Regime Detection")
                st.caption("Unsupervised regime clustering (GaussianHMM) partitioning price history into Bull, Sideways/Volatile, and Bear states.")
                
                with st.spinner("Fitting HMM clustering..."):
                    reg_df, hmm_err = detect_market_regimes(bk_hist)
                
                if hmm_err:
                    st.info(f"HMM offline: {hmm_err}")
                else:
                    # Plotly Close price chart color-coded by regime
                    fig_reg = go.Figure()
                    
                    colors_map = {0: "#f87171", 1: "#eab308", 2: "#4ade80"}  # Bear: red, Sideways: yellow, Bull: green
                    regime_labels = {0: "Bear", 1: "Sideways/Volatile", 2: "Bull"}
                    
                    # Group consecutive regimes to avoid cluttered chart traces
                    # Draw scatter line segments for each regime group
                    for r_val in [0, 1, 2]:
                        subset = reg_df.copy()
                        # Mask out elements that are NOT of this regime to plot them separately
                        subset.loc[subset['Regime'] != r_val, 'Close'] = np.nan
                        fig_reg.add_trace(go.Scatter(
                            x=subset.index, y=subset['Close'],
                            name=regime_labels[r_val],
                            line=dict(color=colors_map[r_val], width=2),
                            connectgaps=False
                        ))
                    
                    fig_reg.update_layout(
                        template="plotly_dark", height=320,
                        paper_bgcolor="#09090b", plot_bgcolor="#09090b",
                        margin=dict(l=0,r=0,t=10,b=0),
                        title=f"{bk_symbol} Price History Colored by HMM Regime",
                        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0)
                    )
                    st.plotly_chart(fig_reg, width='stretch')

                    # Print current regime summary
                    cur_reg = reg_df['Regime_Name'].iloc[-1]
                    cur_reg_val = reg_df['Regime'].iloc[-1]
                    badge_color = "red" if cur_reg_val == 0 else "blue" if cur_reg_val == 1 else "green"
                    st.markdown(f"**Current Market State:** <span class='badge-{badge_color}'>{cur_reg.upper()}</span>", unsafe_allow_html=True)
                    
                    # transition matrix summary
                    st.markdown("""
                    *State transitions are computed automatically on a rolling 1-year sequence of daily log returns and close-to-close volatility.*
                    """)
        except Exception as e:
            st.error(f"Backtesting error: {e}")

# ── TAB: MARKET DATA ────────────────────────────────────────────────────────
with tab_market:
    st.markdown("### 🌐 Global Market Data")

    @st.cache_data(ttl=300)
    def fetch_price(sym):
        try:
            h = yf.Ticker(sym).history(period="2d")
            if len(h) < 2: return None, None
            p, prev = float(h['Close'].iloc[-1]), float(h['Close'].iloc[-2])
            return p, (p - prev) / prev * 100
        except: return None, None

    # ── Forex ──────────────────────────────────────────────────────────────
    st.markdown("#### 💱 Live Forex Rates")
    fx_syms = {"USD/INR": "USDINR=X", "EUR/INR": "EURINR=X", "GBP/INR": "GBPINR=X",
               "USD/EUR": "USDEUR=X", "USD/JPY": "USDJPY=X", "USD/GBP": "USDGBP=X"}
    fx_cols = st.columns(len(fx_syms))
    for col, (name, sym) in zip(fx_cols, fx_syms.items()):
        p, chg = fetch_price(sym)
        col.metric(name, f"{p:.4f}" if p else "N/A", f"{chg:+.2f}%" if chg else "")

    st.markdown("---")

    # ── Commodities ────────────────────────────────────────────────────────
    st.markdown("#### 🪙 Commodities")
    comm_syms = {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F",
                 "Natural Gas": "NG=F", "Copper": "HG=F", "Platinum": "PL=F"}
    comm_cols = st.columns(len(comm_syms))
    for col, (name, sym) in zip(comm_cols, comm_syms.items()):
        p, chg = fetch_price(sym)
        col.metric(name, f"${p:.2f}" if p else "N/A", f"{chg:+.2f}%" if chg else "")

    st.markdown("---")

    # ── Crypto Dashboard ───────────────────────────────────────────────────
    st.markdown("#### ₿ Crypto Dashboard")
    crypto_syms = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD",
                   "Solana": "SOL-USD", "BNB": "BNB-USD",
                   "XRP": "XRP-USD", "DOGE": "DOGE-USD"}
    cr_cols = st.columns(len(crypto_syms))
    for col, (name, sym) in zip(cr_cols, crypto_syms.items()):
        p, chg = fetch_price(sym)
        col.metric(name, f"${p:,.2f}" if p else "N/A", f"{chg:+.2f}%" if chg else "")

    st.markdown("---")

    # ── Global Indices ─────────────────────────────────────────────────────
    st.markdown("#### 📊 Global Indices")
    idx_syms = {"Nifty 50": "^NSEI", "Sensex": "^BSESN", "S&P 500": "^GSPC",
                "NASDAQ": "^IXIC", "Dow Jones": "^DJI", "FTSE 100": "^FTSE"}
    idx_cols = st.columns(len(idx_syms))
    for col, (name, sym) in zip(idx_cols, idx_syms.items()):
        p, chg = fetch_price(sym)
        cur = "₹" if name in ["Nifty 50", "Sensex"] else "$"
        col.metric(name, f"{cur}{p:,.0f}" if p else "N/A", f"{chg:+.2f}%" if chg else "")

    st.markdown("---")

    # ── Sector Performance ─────────────────────────────────────────────────
    st.markdown("#### 🏭 Sector Performance (US + India)")
    sectors = {
        "US IT (XLK)": "XLK", "US Banking (XLF)": "XLF", "US Energy (XLE)": "XLE",
        "US Healthcare (XLV)": "XLV", "US Auto (CARZ)": "CARZ",
        "India IT (INFY.NS)": "INFY.NS", "India Bank (HDFCBANK.NS)": "HDFCBANK.NS",
        "India Energy (ONGC.NS)": "ONGC.NS", "India Auto (TATAMOTORS.NS)": "TATAMOTORS.NS",
    }
    sec_data = {}
    for name, sym in sectors.items():
        _, chg = fetch_price(sym)
        if chg is not None:
            sec_data[name] = round(chg, 2)
    if sec_data:
        sec_df = pd.DataFrame(list(sec_data.items()), columns=["Sector", "Change %"])
        sec_df = sec_df.sort_values("Change %", ascending=True)
        colors = ["#4ade80" if v >= 0 else "#f87171" for v in sec_df["Change %"]]
        fig_sec = go.Figure(go.Bar(
            x=sec_df["Change %"], y=sec_df["Sector"],
            orientation="h", marker_color=colors,
            text=[f"{v:+.2f}%" for v in sec_df["Change %"]], textposition="outside"
        ))
        fig_sec.update_layout(
            template="plotly_dark", height=380,
            paper_bgcolor="#09090b", plot_bgcolor="#09090b",
            margin=dict(l=0, r=60, t=10, b=0),
            xaxis=dict(gridcolor="#27272a"), yaxis=dict(gridcolor="#27272a")
        )
        st.plotly_chart(fig_sec, width='stretch')

    st.markdown("---")

    # ── 52-Week High/Low Tracker ───────────────────────────────────────────
    st.markdown("#### 📏 52-Week High/Low Tracker")
    tracker_syms = ["NVDA", "AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS",
                    "INFY.NS", "HDFCBANK.NS", "BTC-USD", "ETH-USD"]
    tracker_rows = []
    for sym in tracker_syms:
        try:
            tk = yf.Ticker(sym)
            h = tk.history(period="1y")
            if h.empty: continue
            cur_p = float(h['Close'].iloc[-1])
            hi52  = float(h['High'].max())
            lo52  = float(h['Low'].min())
            pct_from_hi = (cur_p - hi52) / hi52 * 100
            pct_from_lo = (cur_p - lo52) / lo52 * 100
            rng = hi52 - lo52
            pos = (cur_p - lo52) / rng * 100 if rng > 0 else 50
            c = get_currency(sym)
            tracker_rows.append({
                "Symbol": sym, "Price": f"{c}{cur_p:,.2f}",
                "52W High": f"{c}{hi52:,.2f}", "From High": f"{pct_from_hi:+.1f}%",
                "52W Low": f"{c}{lo52:,.2f}",  "From Low": f"{pct_from_lo:+.1f}%",
                "Position": f"{pos:.0f}%"
            })
        except: pass
    if tracker_rows:
        st.dataframe(pd.DataFrame(tracker_rows), width='stretch', hide_index=True)

# ── TAB: PORTFOLIO DASHBOARD ─────────────────────────────────────────────────
with tab_portfolio:
    st.markdown("### 💼 Portfolio Dashboard")
    pt_dash = PortfolioTool()

    try:
        holdings_raw = json.loads(pt_dash.get_portfolio_balance())
        all_logs     = PortfolioTool.get_trade_log(500)

        rows, total_val, total_cost = [], 0, 0
        pie_labels, pie_vals = [], []

        for sym, d in holdings_raw.items():
            qty   = d["quantity"]
            price = d["price"]
            val   = d["value"]
            c     = get_currency(sym)
            sym_logs  = [l for l in all_logs if l["symbol"] == sym]
            bought_qty = sum(l["quantity"] for l in sym_logs if l["action"] == "buy")
            sold_qty   = sum(l["quantity"] for l in sym_logs if l["action"] == "sell")
            cost = (bought_qty - sold_qty) * price if isinstance(price, float) else 0
            pnl  = (val - cost) if isinstance(val, float) and cost else 0
            pnl_pct = (pnl / cost * 100) if cost else 0
            if isinstance(val, float):
                total_val  += val
                total_cost += cost
                pie_labels.append(sym)
                pie_vals.append(val)
            rows.append({
                "Symbol": sym, "Qty": f"{qty:.4f}",
                "Price": f"{c}{price:,.2f}" if isinstance(price, float) else price,
                "Value": f"{c}{val:,.2f}"  if isinstance(val,   float) else val,
                "Cost": f"{c}{cost:,.2f}",
                "P&L": f"{c}{pnl:+,.2f}",
                "P&L %": f"{pnl_pct:+.1f}%"
            })

        total_pnl     = total_val - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

        # Summary metrics
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Total Value",   f"${total_val:,.2f}")
        sm2.metric("Total Cost",    f"${total_cost:,.2f}")
        sm3.metric("Total P&L",     f"${total_pnl:+,.2f}", f"{total_pnl_pct:+.1f}%")
        sm4.metric("Positions",     str(len(rows)))

        st.markdown("---")
        pc1, pc2 = st.columns([1, 1])

        with pc1:
            st.markdown("**📋 Holdings Table**")
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

            # Export CSV
            csv_data = pd.DataFrame(rows).to_csv(index=False)
            st.download_button("⬇️ Export Holdings CSV", csv_data,
                               file_name="portfolio.csv", mime="text/csv")

            # Export trade log
            if all_logs:
                log_df  = pd.DataFrame(all_logs)
                log_csv = log_df.to_csv(index=False)
                st.download_button("⬇️ Export Trade Log CSV", log_csv,
                                   file_name="trade_log.csv", mime="text/csv", key="dl_log")

        with pc2:
            st.markdown("**🥧 Allocation Pie Chart**")
            if pie_vals:
                fig_pie = go.Figure(go.Pie(
                    labels=pie_labels, values=pie_vals,
                    hole=0.45,
                    marker=dict(colors=["#00f2ff","#a855f7","#f97316","#4ade80","#f87171","#60a5fa"]),
                    textinfo="label+percent"
                ))
                fig_pie.update_layout(
                    template="plotly_dark", height=340,
                    paper_bgcolor="#09090b",
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=True
                )
                st.plotly_chart(fig_pie, width='stretch')

        st.markdown("---")
        st.markdown("**⚠️ Risk Meter**")
        risk_cols = st.columns(len(holdings_raw))
        for col, (sym, d) in zip(risk_cols, holdings_raw.items()):
            try:
                h = yf.Ticker(sym).history(period="3mo")
                if len(h) > 10:
                    vol = float(h['Close'].pct_change().std() * (252**0.5) * 100)
                    col.metric(f"{sym} Vol", f"{vol:.1f}%",
                               "🔴 High" if vol > 50 else "🟡 Med" if vol > 20 else "🟢 Low")
            except: pass

    except Exception as e:
        st.error(f"Portfolio error: {e}")

# ── TAB: AI SCREENER ─────────────────────────────────────────────────────────
with tab_screener:
    st.markdown("### 🔍 AI Stock Screener")
    st.caption("Scan Indian + US stocks for RSI signals, volume spikes, and near 52W highs/lows")

    sc1, sc2, sc3 = st.columns(3)
    screen_market  = sc1.radio("Market", ["🌍 US", "🇮🇳 India"], horizontal=True, key="sc_mkt")
    rsi_filter     = sc2.selectbox("RSI Filter", ["Oversold (<30)", "Overbought (>70)", "Neutral (30-70)", "All"], key="sc_rsi")
    sort_by        = sc3.selectbox("Sort By", ["RSI", "Change %", "Volume"], key="sc_sort")

    screen_syms = INDIA_SYMBOLS if "India" in screen_market else US_SYMBOLS

    if st.button("⚡ Run Screener", key="sc_run", type="primary"):
        with st.spinner("Scanning stocks..."):
            sc_rows = []
            for sym in screen_syms:
                try:
                    h = yf.Ticker(sym).history(period="3mo")
                    if len(h) < 20: continue
                    delta = h['Close'].diff()
                    gain  = delta.clip(lower=0).rolling(14).mean()
                    loss  = -delta.clip(upper=0).rolling(14).mean()
                    rsi   = float(100 - (100 / (1 + gain.iloc[-1] / max(loss.iloc[-1], 1e-9))))
                    price = float(h['Close'].iloc[-1])
                    prev  = float(h['Close'].iloc[-2])
                    chg   = (price - prev) / prev * 100
                    vol   = int(h['Volume'].iloc[-1])
                    avg_vol = int(h['Volume'].mean())
                    hi52  = float(h['High'].max())
                    lo52  = float(h['Low'].min())
                    c     = get_currency(sym)
                    signal = "🟢 Oversold" if rsi < 30 else "🔴 Overbought" if rsi > 70 else "🟡 Neutral"
                    sc_rows.append({
                        "Symbol": sym, "Price": f"{c}{price:,.2f}",
                        "RSI": round(rsi, 1), "Signal": signal,
                        "Change %": round(chg, 2),
                        "Volume": f"{vol:,}", "Avg Vol": f"{avg_vol:,}",
                        "52W High": f"{c}{hi52:,.2f}", "52W Low": f"{c}{lo52:,.2f}"
                    })
                except: pass

            if sc_rows:
                df_sc = pd.DataFrame(sc_rows)
                if rsi_filter == "Oversold (<30)":    df_sc = df_sc[df_sc["RSI"] < 30]
                elif rsi_filter == "Overbought (>70)": df_sc = df_sc[df_sc["RSI"] > 70]
                elif rsi_filter == "Neutral (30-70)":  df_sc = df_sc[(df_sc["RSI"] >= 30) & (df_sc["RSI"] <= 70)]
                if sort_by == "RSI":       df_sc = df_sc.sort_values("RSI")
                elif sort_by == "Change %": df_sc = df_sc.sort_values("Change %", ascending=False)
                st.dataframe(df_sc, width='stretch', hide_index=True)
                st.download_button("⬇️ Export Screener CSV",
                                   df_sc.to_csv(index=False),
                                   file_name="screener.csv", mime="text/csv", key="dl_sc")

                # AI analysis of top picks
                oversold = df_sc[df_sc["RSI"] < 35]["Symbol"].tolist()[:3]
                if oversold:
                    st.markdown("---")
                    st.markdown("**🤖 AI Analysis of Top Oversold Picks**")
                    with st.spinner("AI analyzing..."):
                        from financial_agent import run_agent_with_retry
                        resp = run_agent_with_retry(
                            agent_team,
                            f"Analyze these oversold stocks briefly: {', '.join(oversold)}. "
                            f"2 bullets per stock max. Are they worth buying?"
                        )
                        if resp and resp.content:
                            st.markdown(resp.content)
            else:
                st.info("No stocks matched the filter.")

# ── TAB: AGENT DEBATE ────────────────────────────────────────────────────────
with tab_debate:
    st.markdown("### ⚔️ Bull vs Bear Agent Debate")
    st.caption("3 agents debate a stock — Bull argues BUY, Bear argues SELL, Judge gives verdict.")
    debate_topic = st.text_input("Enter stock or topic to debate", "NVDA", key="debate_input")
    if st.button("⚡ Start Debate", key="debate_btn"):
        with st.status("Agents are debating...", expanded=True) as status:
            try:
                st.write("Bull and Bear agents are preparing arguments...")
                result = run_debate(debate_topic)
                st.write("Judge is weighng the evidence...")
                col_b, col_r = st.columns(2)
                with col_b:
                    st.markdown("""<div class='card'><div class='card-title'>🟢 Bull Agent</div>""", unsafe_allow_html=True)
                    st.markdown(result["bull"])
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_r:
                    st.markdown("""<div class='card'><div class='card-title'>🔴 Bear Agent</div>""", unsafe_allow_html=True)
                    st.markdown(result["bear"])
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("""<div class='card' style='border-color:#a855f7;'><div class='card-title'>⚖️ Judge Verdict</div>""", unsafe_allow_html=True)
                st.markdown(result["verdict"])
                st.markdown("</div>", unsafe_allow_html=True)
                status.update(label="Debate complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Debate failed", state="error")
                st.error(f"Debate error: {e}")



# ── TAB: AUTO TRADING ────────────────────────────────────────────────────────
with tab_trading:
    st.markdown("### 📊 Autonomous Trading Dashboard")
    pt = PortfolioTool()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**💼 Current Holdings**")
        try:
            holdings = json.loads(pt.get_portfolio_balance())
            for sym, data in holdings.items():
                st.markdown(f"""
                <div class='card' style='padding:12px; margin-bottom:8px;'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='font-weight:600;'>{sym}</span>
                        <span class='badge-blue'>{data['quantity']} units</span>
                    </div>
                    <div style='color:#71717a; font-size:0.85rem; margin-top:4px;'>
                        Price: ${data['price']} &nbsp;|&nbsp; Value: ${data['value']}
                    </div>
                </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"{e}")

    with col2:
        st.markdown("**🤖 Autonomous Rebalance**")
        st.caption("Agents analyze RSI signals and auto-execute trades.")
        if st.button("⚡ Run Auto Rebalance", key="rebalance_btn"):
            with st.spinner("Agents analyzing and trading..."):
                result = pt.autonomous_rebalance()
                st.success(result)

        st.markdown("**📋 Trade Log**")
        logs = PortfolioTool.get_trade_log(10)
        if logs:
            for log in logs:
                badge = "badge-green" if log['action'] == 'buy' else "badge-red"
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #27272a;'>
                    <span style='font-size:0.8rem; color:#71717a;'>{log['ts'][:16]}</span>
                    <span style='font-size:0.85rem;'>{log['symbol']}</span>
                    <span class='{badge}'>{log['action'].upper()}</span>
                    <span style='font-size:0.8rem; color:#a1a1aa;'>{log['quantity']:.4f}</span>
                </div>""", unsafe_allow_html=True)
            if logs:
                st.caption(f"Reason: {logs[0]['reason']}")
        else:
            st.caption("No trades yet.")

# ── TAB: SCHEDULED BRIEFS ────────────────────────────────────────────────────
with tab_schedule:
    st.markdown("### ⏰ Scheduled Morning Briefs")
    st.caption("Agents autonomously run a full market analysis every morning.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""<div class='card'>
            <div class='card-title'>⏰ Schedule</div>
            <div style='font-size:0.9rem; margin-top:8px;'>Every day at <b>09:00 AM</b></div>
            <div style='font-size:0.8rem; color:#71717a; margin-top:4px;'>Covers: News · Portfolio · RSI · Sentiment</div>
        </div>""", unsafe_allow_html=True)
        if st.button("▶️ Run Now", key="brief_btn"):
            with st.status("Running morning brief...", expanded=True) as status:
                st.write("Collecting market news and portfolio data...")
                result = run_morning_brief()
                status.update(label="Brief complete!", state="complete", expanded=False)
                st.success("Brief generated successfully.")
                st.markdown(result)
    with col2:
        st.markdown("**📚 Past Briefs**")
        briefs = get_morning_briefs(5)
        if briefs:
            for b in briefs:
                with st.expander(f"📅 {b['ts'][:16]}"):
                    st.markdown(b['content'])
        else:
            st.caption("No briefs yet. Click 'Run Now' to generate the first one.")

# ── TAB: AI CHAT ─────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#71717a; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>🤖 AI Collective Chat</div>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "suggested_prompt" not in st.session_state:
        st.session_state.suggested_prompt = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "Step" in message["content"]:
                parts = message["content"].split("\n")
                for part in parts:
                    if part.startswith("Step"):
                        st.markdown(f'<div class="reasoning-step">{part}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(part)
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask the Collective...")
    if st.session_state.suggested_prompt:
        prompt = st.session_state.suggested_prompt
        st.session_state.suggested_prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agents thinking..."):
                try:
                    from financial_agent import run_agent_with_retry
                    resp = run_agent_with_retry(agent_team, prompt)
                    full_response = resp.content if resp and resp.content else ""
                    if full_response:
                        st.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        st.error("⚠️ No response. Try rephrasing your question.")
                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)[:300]}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ── TAB: RAG RESEARCH ────────────────────────────────────────────────────────
with tab_rag:
    st.markdown("### 🧠 Textbook Financial RAG Research Assistant")
    st.caption("Perform semantic search over the local 150-page 'Stock Market Master Guide' PDF.")

    rag_query = st.text_input("Ask any financial question (e.g., 'What is standard deviation?' or 'Explain RSI'):")
    
    # Suggestion buttons
    sug_rag_cols = st.columns(3)
    sug_queries = ["What is risk management?", "Explain Bollinger Bands", "What are stock market regimes?"]
    for idx, sq in enumerate(sug_queries):
        if sug_rag_cols[idx].button(sq, key=f"sug_rag_{idx}"):
            rag_query = sq
            
    if rag_query:
        with st.spinner("Retrieving document chunks & synthesizing response..."):
            rag_results = rag_assistant.query(rag_query, k=3)
            
        if not rag_results:
            st.info("No matching information found.")
        else:
            # We display retrieved chunks in tabs or expanders
            st.markdown("#### 📖 Retrieved Guide Chunks")
            for i, r in enumerate(rag_results):
                with st.expander(f"Reference {i+1} — Page {r['page']} (Similarity: {r.get('score', 0):.2f})"):
                    st.markdown(f"*{r['text']}*")
            
            # Now let the AI synthesize an answer using the retrieved context
            st.markdown("#### 🤖 AI Synthesized Explanation")
            with st.spinner("Lead Orchestrator synthesizing..."):
                context_str = "\n\n".join([f"Context (Page {r['page']}): {r['text']}" for r in rag_results])
                prompt = f"Question: {rag_query}\n\nContext from Guide:\n{context_str}\n\nProvide a textbook explanation answering the question based ONLY on the context provided, citing page numbers."
                try:
                    resp = run_agent_with_retry(agent_team, prompt)
                    st.markdown(resp.content if resp and resp.content else "No response generated.")
                except Exception as e:
                    st.error(f"Synthesis error: {e}")


# ── TAB: EARNINGS ANALYZER ──────────────────────────────────────────────────
with tab_earnings:
    st.markdown("### 🎙️ Earnings Call Sentiment Analyzer")
    st.caption("Scrape recent news, reports and transcript summaries to analyze management tone, growth signals, and risks.")

    ern_symbol = st.selectbox("Select Stock for Earnings Analysis:", ["NVDA", "AAPL", "RELIANCE.NS", "TCS.NS"], key="ern_sym")
    ern_btn = st.button("Analyze Recent Earnings", key="ern_run_btn", type="primary")

    if ern_symbol or ern_btn:
        with st.spinner(f"Analyzing {ern_symbol} earnings signals..."):
            # Use search agent to pull summaries
            from financial_agent import run_agent_with_retry, web_search_agent
            search_query = f"{ern_symbol} recent earnings call transcript summary growth highlights risk outlook"
            resp = run_agent_with_retry(web_search_agent, search_query)
            summary_content = resp.content if resp and resp.content else ""
            
        if not summary_content:
            st.info("Could not fetch earnings summaries.")
        else:
            # Run FinBERT on each bullet/paragraph of the transcript summary
            sentences = [s.strip() for s in summary_content.split("\n") if len(s.strip()) > 10]
            
            # Compile scores
            positives, negatives, neutrals = [], [], []
            sentiment_details = []
            
            for s in sentences[:8]: # limit to 8 for speed
                pos, neg, neu = finbert_analyzer.analyze(s)
                positives.append(pos)
                negatives.append(neg)
                neutrals.append(neu)
                
                label = "🟢 Positive" if pos > neg and pos > neu else "🔴 Negative" if neg > pos and neg > neu else "🟡 Neutral"
                sentiment_details.append({"text": s, "sentiment": label, "pos": pos, "neg": neg, "neu": neu})
                
            avg_pos = np.mean(positives) if positives else 0.0
            avg_neg = np.mean(negatives) if negatives else 0.0
            avg_neu = np.mean(neutrals) if neutrals else 1.0
            
            # Bullish score out of 10
            bullish_score = (avg_pos * 10) + (avg_neu * 2)
            bullish_score = min(max(bullish_score, 0.5), 9.9)
            
            st.markdown("#### 🎯 FinBERT Sentiment & Tone Metrics")
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Bullish Score", f"{bullish_score:.1f} / 10", "🟢 Bullish" if bullish_score > 6 else "🔴 Bearish" if bullish_score < 4 else "🟡 Neutral")
            ec2.metric("Avg Positive Probability", f"{avg_pos*100:.1f}%")
            ec3.metric("Avg Negative Probability", f"{avg_neg*100:.1f}%")
            
            # Print highlights
            st.markdown("#### 📋 Segment-by-Segment Sentiment Analysis")
            for sd in sentiment_details:
                label_color = "green" if "Positive" in sd["sentiment"] else "red" if "Negative" in sd["sentiment"] else "blue"
                st.markdown(f"""
                <div class='card' style='padding:10px; margin-bottom:8px;'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='font-size:0.9rem;'>{sd['text']}</span>
                        <span class='badge-{label_color}'>{sd['sentiment'].upper()}</span>
                    </div>
                </div>""", unsafe_allow_html=True)


# ── TAB: RL OPTIMIZATION ─────────────────────────────────────────────────────
with tab_rl:
    st.markdown("### 🤖 Reinforcement Learning Portfolio Optimizer")
    st.caption("Train a Policy Gradient agent in real-time to optimize asset weights dynamically over historical data.")

    rl_episodes = st.slider("RL Training Episodes", min_value=5, max_value=50, value=20, step=5)
    rl_run = st.button("Train RL Portfolio Agent", key="rl_run_btn", type="primary")

    # Load 1y daily prices for NVDA, AAPL, BTC-USD
    @st.cache_data(ttl=600)
    def load_rl_matrix():
        tickers = ["NVDA", "AAPL", "BTC-USD"]
        df_list = []
        for t in tickers:
            hist = yf.Ticker(t).history(period="1y")['Close']
            df_list.append(hist)
        merged = pd.concat(df_list, axis=1)
        merged.columns = tickers
        merged.ffill(inplace=True)
        merged.dropna(inplace=True)
        return merged

    if rl_run:
        price_matrix_df = load_rl_matrix()
        if price_matrix_df.empty:
            st.warning("Could not load asset price matrix.")
        else:
            st.markdown("#### ⏳ Running Reinforcement Learning (PG Policy) Training...")
            progress_bar = st.progress(0.0)
            
            agent = RLOptimizerAgent(state_dim=6, action_dim=3)
            price_matrix = price_matrix_df.values
            
            episodes_rewards = []
            
            # We run episode loop, updating Streamlit progress
            for ep in range(rl_episodes):
                rewards, weights = agent.train_on_history(price_matrix, episodes=1)
                episodes_rewards.append(rewards[0])
                progress_bar.progress((ep + 1) / rl_episodes)
                
            st.success("RL Portfolio Agent Training Completed!")
            
            # Plot training curve (Episode vs total reward)
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(y=episodes_rewards, mode="lines+markers", name="Episode Cumulative Reward", line=dict(color="#a855f7", width=2)))
            fig_reward.update_layout(
                template="plotly_dark", height=240, paper_bgcolor="#09090b", plot_bgcolor="#09090b",
                margin=dict(l=0,r=0,t=10,b=0), title="RL Training Rewards History (Policy Gradient)"
            )
            st.plotly_chart(fig_reward, width='stretch')
            
            # Compute portfolio cumulative returns using final RL weights vs Equal-weight
            final_weights = agent.select_action( (price_matrix[1:] / price_matrix[:-1] - 1)[-2:].flatten() )
            final_weights = final_weights / np.sum(final_weights) # normalize
            
            daily_returns = price_matrix_df.pct_change().dropna()
            rl_daily_returns = (daily_returns * final_weights).sum(axis=1)
            eq_daily_returns = daily_returns.mean(axis=1)
            
            cum_rl = (1 + rl_daily_returns).cumprod()
            cum_eq = (1 + eq_daily_returns).cumprod()
            
            st.markdown("#### ⚖️ RL Portfolio Allocations vs Equal Weight")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=cum_rl.index, y=cum_rl, name="RL Optimized Weight", line=dict(color="#4ade80", width=2.5)))
            fig_comp.add_trace(go.Scatter(x=cum_eq.index, y=cum_eq, name="Equal Weight Portfolio", line=dict(color="#71717a", width=1.5, dash="dot")))
            fig_comp.update_layout(template="plotly_dark", height=280, paper_bgcolor="#09090b", plot_bgcolor="#09090b", margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_comp, width='stretch')
            
            # Allocation Display
            aw1, aw2, aw3 = st.columns(3)
            assets = ["NVDA", "AAPL", "BTC-USD"]
            aw1.metric("NVDA Weight", f"{final_weights[0]*100:.1f}%")
            aw2.metric("AAPL Weight", f"{final_weights[1]*100:.1f}%")
            aw3.metric("BTC-USD Weight", f"{final_weights[2]*100:.1f}%")
    else:
        st.info("Click 'Train RL Portfolio Agent' to train policy gradient weights in real-time.")

