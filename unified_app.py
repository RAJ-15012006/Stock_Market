import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from financial_agent import agent_team, run_debate, run_morning_brief, get_morning_briefs, PortfolioTool, set_model, MODEL as DEFAULT_MODEL
from phi.model.groq import Groq
import threading
import json
import os

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
        &nbsp;&nbsp;&nbsp;
        🟢 NVDA $875.40 +2.1%&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        🔴 AAPL $189.30 -0.5%&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        🟢 BTC $63,200 +1.2%&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        🟢 TSLA $177.80 +0.8%&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        🟢 ETH $3,100 +0.5%&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        🔴 META $485.20 -0.3%&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        🟢 MSFT $415.60 +1.1%&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        🟢 GOOGL $172.40 +0.9%&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        🔴 AMZN $182.10 -0.2%&nbsp;&nbsp;&nbsp;
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
    "Latest crypto news",
    "Compare AAPL vs MSFT",
    "Market sentiment today"
]
for i, (col, s) in enumerate(zip(sugg_cols, suggestions)):
    with col:
        if st.button(s, key=f"sugg_{i}", use_container_width=True):
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

DEFAULT_SYMBOLS = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "BTC-USD", "ETH-USD", "NFLX"]

# Sidebar: Market Dashboard
with st.sidebar:
    st.image(r"C:\Users\rajku\.gemini\antigravity\brain\a3115520-e6e6-4487-9960-fca77739e19e\financial_ai_banner_1778081471404.png", use_container_width=True)
    st.header("📈 Market Pulse")
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    
    model_options = {
        "Llama 3.1 8B (Fast & Reliable)": "llama-3.1-8b-instant",
        "Llama 3.3 70B (Powerful)": "llama-3.3-70b-versatile",
        "Gemini 2.0 Flash (Advanced Backup)": "gemini-2.0-flash",
        "Qwen 2.5 Coder 7B (Local)": "qwen2.5-coder:7b",
        "DeepSeek R1 1.5B (Local)": "deepseek-r1:1.5b",
    }
    
    selected_model_name = st.selectbox("Select Agent Model", list(model_options.keys()), index=0)
    selected_model_id = model_options[selected_model_name]
    
    if "current_model_id" not in st.session_state or st.session_state.current_model_id != selected_model_id:
        set_model(selected_model_id)
        st.session_state.current_model_id = selected_model_id
        st.session_state.current_model_name = selected_model_name.split(" (")[0]
        st.session_state.current_model_tier = "70B · Versatile" if "70B" in selected_model_name else "8B · Instant" if "8B" in selected_model_name else "SMoE · 32k"
        st.toast(f"Switched to {selected_model_name}")

    st.markdown("---")
    st.subheader("🔍 Symbol Selection")
    symbol = st.selectbox("Quick Select", DEFAULT_SYMBOLS)

    custom = st.text_input("Or type custom symbol", "").upper().strip()
    if custom:
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
                st.metric("Price", f"${price:.2f}", f"{change:+.2f}%")

                col1, col2 = st.columns(2)
                col1.metric("High", f"${hist['High'].max():.2f}")
                col2.metric("Low", f"${hist['Low'].min():.2f}")
                col1.metric("Volume", f"{int(hist['Volume'].iloc[-1]):,}")
                col2.metric("Avg Vol", f"{int(hist['Volume'].mean()):,}")

                try:
                    mc = info.get('market_cap')
                    yh = info.get('year_high')
                    col1.metric("Market Cap", f"${mc/1e9:.1f}B" if mc else "N/A")
                    col2.metric("52W High", f"${yh:.2f}" if yh else "N/A")
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
                st.plotly_chart(fig, use_container_width=True)

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
tab_chat, tab_debate, tab_trading, tab_schedule = st.tabs([
    "🤖 AI Chat", "⚔️ Agent Debate", "📊 Auto Trading", "⏰ Scheduled Briefs"
])

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
            try:
                # Use a placeholder for the streaming response
                response_placeholder = st.empty()
                full_response = ""
                
                # Run the agent in streaming mode
                # Run with streaming retries and automatic 8B fallback
                import time
                max_chat_retries = 3
                fallback_model_id = "llama-3.1-8b-instant"
                fallbacks = [("Llama 3.1 8B", "llama-3.1-8b-instant", "8B · Instant"), 
                             ("Llama 3.1 70B", "llama-3.1-70b-versatile", "70B · Versatile"),
                             ("Llama 3.3 70B", "llama-3.3-70b-versatile", "70B · Heavyweight")]
                
                for attempt in range(max_chat_retries):
                    try:
                        for chunk in agent_team.run(prompt, stream=True):
                            if chunk.content:
                                full_response += chunk.content
                                response_placeholder.markdown(full_response + "▌")
                        break
                    except Exception as e:
                        err_msg = str(e)
                        if "429" in err_msg:
                            # TPD or all retries failed: trigger fallback chain
                            is_tpd = "tokens per day" in err_msg.lower() or "tpd" in err_msg.lower() or "requests per day" in err_msg.lower()
                            
                            if is_tpd or attempt == max_chat_retries - 1:
                                found_fallback = False
                                for name, mid, tier in fallbacks:
                                    if not agent_team.model or not hasattr(agent_team.model, "id") or agent_team.model.id != mid:
                                        response_placeholder.warning(f"Rate limit reached. Switching to {name} for this session...")
                                        set_model(mid)
                                        st.session_state.current_model_id = mid
                                        st.session_state.current_model_name = name
                                        st.session_state.current_model_tier = tier
                                        
                                        full_response = ""
                                        try:
                                            for chunk in agent_team.run(prompt, stream=True):
                                                if chunk.content:
                                                    full_response += chunk.content
                                                    response_placeholder.markdown(full_response + "▌")
                                            found_fallback = True
                                            break
                                        except: continue
                                if found_fallback: break
                            
                            time.sleep((attempt + 1) * 3)
                            full_response = "" 
                            continue
                        raise e
                
                # Final render with styling
                response_placeholder.empty()
                if full_response:
                    if "Step" in full_response:
                        parts = full_response.split("\n")
                        for part in parts:
                            if part.startswith("Step"):
                                st.markdown(f'<div class="reasoning-step">{part}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(part)
                    else:
                        st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("⚠️ All models reached their daily limit. Please try again tomorrow or upgrade your Groq tier.")
                
            except Exception as e:
                error_msg = f"⚠️ Agent error: {str(e)[:200]}. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

