import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from financial_agent import agent_team
import os

# Page configuration
st.set_page_config(
    page_title="Intelligent Financial Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        color: #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #1f77b4;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Banner Image
banner_path = r"C:\Users\rajku\.gemini\antigravity\brain\a3115520-e6e6-4487-9960-fca77739e19e\financial_ai_banner_1778081471404.png"
if os.path.exists(banner_path):
    st.image(banner_path, width='stretch')

# Title and Description
st.title("🚀 Intelligent Financial AI Assistant")
st.markdown("---")

# Caching stock data
@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    return stock.info, stock.history(period="1mo")

# Sidebar
with st.sidebar:
    st.header("📊 Market Dashboard")
    symbol = st.text_input("Enter Stock Symbol", value="NVDA").upper()
    
    if symbol:
        try:
            with st.spinner(f"Fetching data for {symbol}..."):
                info, hist = get_stock_data(symbol)
                try:
                    st.metric("Price", f"${info.get('currentPrice', 'N/A')}")
                    st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                except Exception:
                    pass
                
                # Plotly Chart
                fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                open=hist['Open'],
                                high=hist['High'],
                                low=hist['Low'],
                                close=hist['Close'])])
                fig.update_layout(
                    title=f"{symbol} Trend", 
                    template="plotly_dark", 
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=300
                )
                st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.header("⚙️ Agent Memory")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me about stocks, market trends, or sentiment analysis..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        with st.spinner("Our AI agents are analyzing and planning your request..."):
            try:
                # The agent_team handles the task planning and multi-agent coordination
                response = agent_team.run(prompt)
                full_response = response.content
                response_container.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Agent Error: {e}")
