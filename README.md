# 🤖 Agentic AI Financial Intelligence Platform

A full-featured AI-powered stock market dashboard with multi-agent analysis, real-time data, Indian + US markets, and autonomous trading simulation.

## 🌟 Features

| Tab | What it does |
|-----|-------------|
| 🤖 AI Chat | Ask anything — agents answer with live data |
| 🕯️ Candle Trading | 8 chart types · RSI · MACD · BB · Buy/Sell signals |
| 🌐 Market Data | Forex · Commodities · Crypto · Global Indices · Sectors · 52W tracker |
| 💼 Portfolio | P&L · Pie chart · Risk meter · CSV export |
| 🔍 AI Screener | Scan Indian + US stocks for RSI/volume signals |
| ⚔️ Agent Debate | Bull vs Bear vs Judge — 3-agent stock debate |
| 📊 Auto Trading | RSI-based autonomous rebalancing |
| ⏰ Morning Brief | AI-generated daily market brief saved to DB |

## 🇮🇳 Indian Stocks Supported
RELIANCE, TCS, INFY, HDFCBANK, WIPRO, ICICIBANK, BAJFINANCE, ADANIENT, SBIN, TATAMOTORS, HINDUNILVR, MARUTI, SUNPHARMA, ONGC, AXISBANK

## 🛠️ Tech Stack
- **AI**: Agno framework · Groq (Llama 3.1) · Gemini 2.0 Flash fallback
- **Data**: YFinance · DuckDuckGo Search
- **UI**: Streamlit · Plotly · Three.js globe
- **Storage**: SQLite (sessions + portfolio + briefs)

## 🚀 Run Locally

```bash
# 1. Clone
git clone <your-repo-url>
cd agentic_ai_stock

# 2. Install
pip install -r requirements.txt

# 3. Add API keys
cp .env.example .env
# Edit .env → add GROQ_API_KEY and GOOGLE_API_KEY

# 4. Run
python -m streamlit run unified_app.py
```

Open → http://localhost:8501

## 🐳 Docker

```bash
docker-compose up --build
```

Open → http://localhost:8501

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push to GitHub (make sure `.env` is in `.gitignore`)
2. Go to https://share.streamlit.io
3. Click **New app** → select your repo → set `unified_app.py` as main file
4. Add secrets: `GROQ_API_KEY` and `GOOGLE_API_KEY` in the Secrets panel
5. Click **Deploy** ✅

## 🔑 Environment Variables

```env
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
```

Get keys:
- Groq (free): https://console.groq.com
- Google (free): https://aistudio.google.com

## 📄 License
MIT
