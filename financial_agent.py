from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.db.sqlite import SqliteDb
from agno.tools import Toolkit
from dotenv import load_dotenv
import yfinance as yf
import json, os, sqlite3, time
from datetime import datetime

load_dotenv()

MODEL = "llama-3.1-8b-instant"
agent_storage = "agent_sessions.db"
storage = SqliteDb(db_file=agent_storage, session_table="financial_assistant")


def run_agent_with_retry(agent, prompt, max_retries=4):
    """Run agent with Groq → Gemini fallback on rate limit."""
    last_err = None
    for i in range(max_retries):
        try:
            return agent.run(prompt, stream=False)
        except Exception as e:
            last_err = e
            err = str(e).lower()
            if "429" in err or "rate limit" in err:
                # try gemini fallback
                try:
                    agent.model = Gemini(id="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
                    return agent.run(prompt, stream=False)
                except Exception:
                    agent.model = Groq(id="llama-3.1-8b-instant")
                time.sleep(2 * (i + 1))
            else:
                raise e
    raise last_err


def delegate_to_finance_agent(task: str, symbol: str = "None") -> str:
    """Get financial data, price, fundamentals for a stock symbol."""
    try:
        return run_agent_with_retry(finance_agent, f"Task: {task}. Symbol: {symbol}").content
    except Exception as e:
        return f"Finance error: {str(e)}"


def delegate_to_web_search_agent(task: str, symbol: str = "None") -> str:
    """Search latest news and market updates for a topic or symbol."""
    try:
        return run_agent_with_retry(web_search_agent, f"Task: {task}. Symbol: {symbol}").content
    except Exception as e:
        return f"Search error: {str(e)}"


def delegate_to_sentiment_agent(task: str, symbol: str = "None") -> str:
    """Get market sentiment score and mood for a symbol."""
    try:
        return run_agent_with_retry(sentiment_agent, f"Task: {task}. Symbol: {symbol}").content
    except Exception as e:
        return f"Sentiment error: {str(e)}"


def set_model(model_id: str):
    global MODEL
    MODEL = model_id
    if "gemini" in MODEL.lower():
        new_model = Gemini(id=MODEL, api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        new_model = Groq(id=MODEL)
    for ag in [web_search_agent, finance_agent, sentiment_agent,
               agent_team, bull_agent, bear_agent, judge_agent]:
        ag.model = new_model
    return f"Model switched to {model_id}"


# ── Portfolio Tool ────────────────────────────────────────────────────────────
class PortfolioTool(Toolkit):
    DB = "portfolio.db"

    def __init__(self):
        self._init_db()
        super().__init__(name="portfolio_tool", tools=[
            self.get_portfolio_balance,
            self.simulate_trade,
            self.autonomous_rebalance,
        ])

    def _init_db(self):
        con = sqlite3.connect(self.DB)
        con.execute("CREATE TABLE IF NOT EXISTS holdings (symbol TEXT PRIMARY KEY, quantity REAL)")
        con.execute("""CREATE TABLE IF NOT EXISTS trade_log
                       (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT, symbol TEXT, action TEXT, quantity REAL, reason TEXT)""")
        if not con.execute("SELECT 1 FROM holdings").fetchone():
            for sym, qty in [("NVDA", 10), ("AAPL", 5), ("BTC-USD", 0.5)]:
                con.execute("INSERT INTO holdings VALUES (?,?)", (sym, qty))
        con.commit()
        con.close()

    def get_portfolio_balance(self) -> str:
        """Returns current portfolio holdings and estimated value."""
        con = sqlite3.connect(self.DB)
        rows = con.execute("SELECT symbol, quantity FROM holdings").fetchall()
        con.close()
        result = {}
        for sym, qty in rows:
            try:
                price = yf.Ticker(sym).history(period="1d")['Close'].iloc[-1]
                result[sym] = {"quantity": qty, "price": round(float(price), 2),
                               "value": round(qty * float(price), 2)}
            except Exception:
                result[sym] = {"quantity": qty, "price": "N/A", "value": "N/A"}
        return json.dumps(result)

    def simulate_trade(self, symbol: str, quantity: float, action: str, reason: str = "Manual") -> str:
        """Executes a simulated buy or sell trade and logs it."""
        con = sqlite3.connect(self.DB)
        row = con.execute("SELECT quantity FROM holdings WHERE symbol=?", (symbol,)).fetchone()
        qty = row[0] if row else 0.0
        qty = qty + quantity if action == "buy" else max(0, qty - quantity)
        con.execute("INSERT OR REPLACE INTO holdings VALUES (?,?)", (symbol, qty))
        con.execute("INSERT INTO trade_log(ts,symbol,action,quantity,reason) VALUES(?,?,?,?,?)",
                    (datetime.now().isoformat(), symbol, action, quantity, reason))
        con.commit()
        con.close()
        return f"✅ {action.upper()} {quantity} {symbol}. New holding: {qty:.4f}. Reason: {reason}"

    def autonomous_rebalance(self) -> str:
        """Rebalances portfolio based on RSI signals."""
        try:
            con = sqlite3.connect(self.DB)
            holdings = con.execute("SELECT symbol, quantity FROM holdings").fetchall()
            con.close()
            actions = []
            for sym, qty in holdings:
                try:
                    hist = yf.Ticker(sym).history(period="1mo")
                    if len(hist) < 14:
                        continue
                    delta = hist['Close'].diff()
                    gain = delta.clip(lower=0).rolling(14).mean()
                    loss = -delta.clip(upper=0).rolling(14).mean()
                    rsi = 100 if loss.iloc[-1] == 0 else 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1]))
                    price = float(hist['Close'].iloc[-1])
                    if rsi < 30:
                        actions.append(self.simulate_trade(sym, round(50 / price, 4), "buy", f"RSI={rsi:.1f} oversold"))
                    elif rsi > 70:
                        sell_qty = round(qty * 0.1, 4)
                        if sell_qty > 0:
                            actions.append(self.simulate_trade(sym, sell_qty, "sell", f"RSI={rsi:.1f} overbought"))
                except Exception as e:
                    actions.append(f"⚠️ {sym}: {str(e)}")
            return "\n".join(actions) if actions else "No rebalancing needed."
        except Exception as e:
            return f"Rebalance error: {str(e)}"

    @staticmethod
    def get_trade_log(limit: int = 10) -> list:
        con = sqlite3.connect(PortfolioTool.DB)
        rows = con.execute(
            "SELECT ts, symbol, action, quantity, reason FROM trade_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        con.close()
        return [{"ts": r[0], "symbol": r[1], "action": r[2], "quantity": r[3], "reason": r[4]} for r in rows]


portfolio_tool = PortfolioTool()

# ── Agents ────────────────────────────────────────────────────────────────────
_STRICT = [
    "Reply in MAX 5 bullet points.",
    "Each bullet: one short sentence.",
    "NO greetings, NO disclaimers, NO filler.",
    "Use tools once. Do not repeat tool calls.",
]

web_search_agent = Agent(
    name="Web Search Agent",
    role="Real-time market news",
    model=Groq(id=MODEL),
    tools=[DuckDuckGoTools()],
    instructions=_STRICT + ["Search once. Return top 3 news bullets only."],
    db=storage, add_history_to_context=True, markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Stock data and portfolio analysis",
    model=Groq(id=MODEL),
    tools=[
        YFinanceTools(enable_stock_price=True, enable_analyst_recommendations=True,
                      enable_stock_fundamentals=True, enable_company_news=True),
        portfolio_tool,
    ],
    instructions=_STRICT + ["Fetch data with tools. Show price, PE, recommendation, RSI in bullets."],
    db=storage, add_history_to_context=True, markdown=True,
)

sentiment_agent = Agent(
    name="Sentiment Agent",
    role="Market mood",
    model=Groq(id=MODEL),
    instructions=["Reply with: Score: X/10 | Mood: WORD | Reason: one sentence. Nothing else."],
    db=storage, add_history_to_context=True, markdown=True,
)

agent_team = Agent(
    name="Lead Orchestrator",
    tools=[
        delegate_to_finance_agent,
        delegate_to_web_search_agent,
        delegate_to_sentiment_agent,
        YFinanceTools(enable_stock_price=True, enable_analyst_recommendations=True,
                      enable_stock_fundamentals=True),
        DuckDuckGoTools(),
    ],
    model=Groq(id=MODEL),
    instructions=[
        "You are a financial AI. Answer the user question using tools.",
        "MAX 5 bullet points. Each bullet = 1 short sentence.",
        "NEVER show raw JSON. Summarize numbers in plain English.",
        "Call each tool ONCE. Do not loop.",
        "End every reply with: '⚠️ Educational only. Not financial advice.'",
    ],
    db=storage, add_history_to_context=True, markdown=True,
)

bull_agent = Agent(
    name="Bull Agent", role="Optimistic analyst",
    model=Groq(id=MODEL), tools=[],
    instructions=["Argue BUY in exactly 4 bullets. Max 120 words total. No tools."],
    markdown=True,
)

bear_agent = Agent(
    name="Bear Agent", role="Pessimistic analyst",
    model=Groq(id=MODEL), tools=[],
    instructions=["Argue SELL in exactly 4 bullets. Max 120 words total. No tools."],
    markdown=True,
)

judge_agent = Agent(
    name="Judge Agent", role="Neutral arbitrator",
    model=Groq(id=MODEL), tools=[],
    instructions=["Verdict: BUY/HOLD/SELL with confidence %. Then 2 sentences of reasoning. No tools."],
    markdown=True,
)


def run_debate(topic: str) -> dict:
    try:
        bull_content = run_agent_with_retry(bull_agent, f"Topic: {topic}").content or "No argument."
        bear_content = run_agent_with_retry(bear_agent, f"Topic: {topic}").content or "No argument."
        verdict = run_agent_with_retry(judge_agent,
                                       f"Bull:\n{bull_content}\n\nBear:\n{bear_content}").content or "No verdict."
        return {"bull": bull_content, "bear": bear_content, "verdict": verdict}
    except Exception as e:
        return {"bull": f"Error: {e}", "bear": f"Error: {e}", "verdict": f"Debate failed: {e}"}


def run_morning_brief() -> str:
    """Generate and SAVE morning brief to DB."""
    try:
        news = run_agent_with_retry(web_search_agent, "Top 3 market news today. 3 bullets max.")
        portfolio = run_agent_with_retry(finance_agent, "Portfolio value and RSI for NVDA, AAPL, BTC-USD. Bullets only.")
        rebalance = portfolio_tool.autonomous_rebalance()

        brief = f"""# ☕ Morning Market Brief — {datetime.now().strftime('%Y-%m-%d %H:%M')}

### 🌍 Global Headlines
{getattr(news, 'content', str(news))}

### 📊 Portfolio & RSI
{getattr(portfolio, 'content', str(portfolio))}

### ⚖️ Rebalance Action
{rebalance}

*⚠️ Educational only. Not financial advice.*"""

        # Save to DB
        con = sqlite3.connect(agent_storage)
        con.execute("CREATE TABLE IF NOT EXISTS morning_briefs "
                    "(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, content TEXT)")
        con.execute("INSERT INTO morning_briefs(ts, content) VALUES(?,?)",
                    (datetime.now().isoformat(), brief))
        con.commit()
        con.close()
        return brief
    except Exception as e:
        return f"Morning Brief Failed: {str(e)}"


def get_morning_briefs(limit: int = 5) -> list:
    try:
        con = sqlite3.connect(agent_storage)
        con.execute("CREATE TABLE IF NOT EXISTS morning_briefs "
                    "(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, content TEXT)")
        rows = con.execute(
            "SELECT ts, content FROM morning_briefs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        con.close()
        return [{"ts": r[0], "content": r[1]} for r in rows]
    except Exception:
        return []
