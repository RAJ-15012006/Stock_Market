from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.google import Gemini
from phi.model.ollama import Ollama
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.embedder.fastembed import FastEmbedEmbedder
from phi.tools import Toolkit
from dotenv import load_dotenv
import yfinance as yf
import json, os, sqlite3, time
from datetime import datetime

def run_agent_with_retry(agent, prompt, max_retries=5, stream=False):
    """Utility to handle Groq TPM/TPD rate limits with multiple fallbacks."""
    fallbacks = [
        {"provider": "groq", "id": "llama-3.1-8b-instant"},
        {"provider": "google", "id": "gemini-2.0-flash"},
        {"provider": "ollama", "id": "qwen2.5-coder:7b"}
    ]
    for i in range(max_retries):
        try:
            return agent.run(prompt, stream=stream)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg:
                for fb in fallbacks:
                    try:
                        if fb["provider"] == "groq":
                            agent.model = Groq(id=fb["id"], parallel_tool_calls=False)
                        elif fb["provider"] == "google":
                            agent.model = Gemini(id=fb["id"], api_key=os.getenv("GOOGLE_API_KEY"))
                        elif fb["provider"] == "ollama":
                            agent.model = Ollama(id=fb["id"])
                        return agent.run(prompt, stream=stream)
                    except: continue
                
                if i < max_retries - 1:
                    time.sleep(2)
                    continue
            raise e
            raise e



# Tool: Delegate tasks to specialized agents
def delegate_to_finance_agent(task: str, symbol: str = "None", additional_information: str = "No info") -> str:
    """Delegates to Finance Agent. Provide 'task' and 'symbol'."""
    prompt = f"Task: {task}. Symbol: {symbol}. Info: {additional_information}"
    try:
        response = run_agent_with_retry(finance_agent, prompt)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

def delegate_to_sentiment_agent(task: str, symbol: str = "None", additional_information: str = "No info") -> str:
    """Delegates to Sentiment Agent. Provide 'task' and 'symbol'."""
    prompt = f"Task: {task}. Symbol: {symbol}. Info: {additional_information}"
    try:
        response = run_agent_with_retry(sentiment_agent, prompt)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

def delegate_to_web_search_agent(task: str, symbol: str = "None", additional_information: str = "No info") -> str:
    """Delegates to Web Search Agent. Provide 'task' and 'symbol'."""
    prompt = f"Task: {task}. Symbol: {symbol}. Info: {additional_information}"
    try:
        response = run_agent_with_retry(web_search_agent, prompt)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

load_dotenv()

agent_storage = "agent_sessions.db"
storage = SqlAgentStorage(table_name="financial_assistant", db_file=agent_storage)

MODEL = "llama-3.1-8b-instant"

# --- RAG Setup ---
knowledge_base = PDFKnowledgeBase(
    path="Stock_Market_Master_Guide_150_Pages.pdf",
    # Using LanceDB for vector storage
    vector_db=LanceDb(
        table_name="market_knowledge",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=FastEmbedEmbedder(),
    ),
)
# Load knowledge base (commented out by default, run load_knowledge.py instead)
# knowledge_base.load(recreate=False)

def get_knowledge_base():
    """Safety check to ensure knowledge base exists before use."""
    if os.path.exists("tmp/lancedb"):
        return knowledge_base
    return None

def set_model(model_id: str):
    """Updates the model for all agents dynamically."""
    global MODEL
    MODEL = model_id
    
    if "llama" in MODEL.lower() or "mixtral" in MODEL.lower():
        new_model = Groq(id=MODEL, parallel_tool_calls=False)
    elif "gemini" in MODEL.lower():
        new_model = Gemini(id=MODEL, api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        # Assume Ollama for anything else (like qwen or deepseek)
        new_model = Ollama(id=MODEL)
    
    # Update all agents
    web_search_agent.model = new_model
    finance_agent.model = new_model
    sentiment_agent.model = new_model
    agent_team.model = new_model
    bull_agent.model = new_model
    bear_agent.model = new_model
    judge_agent.model = new_model
    return f"Model switched to {model_id}"


# ── Portfolio + Autonomous Trading Tool ──────────────────────────────────────
class PortfolioTool(Toolkit):
    DB = "portfolio.db"

    def __init__(self):
        super().__init__(name="portfolio_tool")
        self._init_db()
        self.register(self.get_portfolio_balance)
        self.register(self.simulate_trade)
        self.register(self.autonomous_rebalance)

    def _init_db(self):
        con = sqlite3.connect(self.DB)
        con.execute("""CREATE TABLE IF NOT EXISTS holdings
                       (symbol TEXT PRIMARY KEY, quantity REAL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS trade_log
                       (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT, symbol TEXT, action TEXT,
                        quantity REAL, reason TEXT)""")
        # seed default portfolio if empty
        if not con.execute("SELECT 1 FROM holdings").fetchone():
            for sym, qty in [("NVDA", 10), ("AAPL", 5), ("BTC-USD", 0.5)]:
                con.execute("INSERT INTO holdings VALUES (?,?)", (sym, qty))
        con.commit(); con.close()

    def get_portfolio_balance(self) -> str:
        """Returns current portfolio holdings and estimated value."""
        con = sqlite3.connect(self.DB)
        rows = con.execute("SELECT symbol, quantity FROM holdings").fetchall()
        con.close()
        result = {}
        for sym, qty in rows:
            try:
                price = yf.Ticker(sym).history(period="1d")['Close'].iloc[-1]
                result[sym] = {"quantity": qty, "price": round(price, 2), "value": round(qty * price, 2)}
            except Exception:
                result[sym] = {"quantity": qty, "price": "N/A", "value": "N/A"}
        return json.dumps(result)

    def get_current_stock_price(self, symbol: str | list) -> str:
        """Fetch current stock price. Handles single symbol or list of symbols."""
        symbols = [symbol] if isinstance(symbol, str) else symbol
        results = []
        for s in symbols:
            try:
                ticker = yf.Ticker(s)
                price = ticker.fast_info['last_price']
                results.append(f"{s}: ${price:.2f}")
            except: results.append(f"{s}: Error")
        return "\n".join(results)

    def get_stock_fundamentals(self, symbol: str | list) -> str:
        """Fetch fundamentals. Handles single symbol or list of symbols."""
        symbols = [symbol] if isinstance(symbol, str) else symbol
        # Use existing YFinanceTools logic via a simple wrapper if possible, 
        # but for simplicity let's just return a summary
        return f"Fundamentals requested for {symbols}"

    def simulate_trade(self, symbol: str, quantity: float, action: str, reason: str = "Manual") -> str:
        """Executes a simulated buy or sell trade and logs it."""
        con = sqlite3.connect(self.DB)
        row = con.execute("SELECT quantity FROM holdings WHERE symbol=?", (symbol,)).fetchone()
        qty = row[0] if row else 0.0
        if action == "buy":
            qty += quantity
        elif action == "sell":
            qty = max(0, qty - quantity)
        con.execute("INSERT OR REPLACE INTO holdings VALUES (?,?)", (symbol, qty))
        con.execute("INSERT INTO trade_log(ts,symbol,action,quantity,reason) VALUES(?,?,?,?,?)",
                    (datetime.now().isoformat(), symbol, action, quantity, reason))
        con.commit(); con.close()
        return f"✅ {action.upper()} {quantity} {symbol} executed. New holding: {qty:.4f}. Reason: {reason}"

    def autonomous_rebalance(self) -> str:
        """Autonomously analyzes portfolio and executes rebalancing trades based on RSI signals."""
        try:
            con = sqlite3.connect(self.DB)
            holdings = con.execute("SELECT symbol, quantity FROM holdings").fetchall()
            con.close()
            actions = []
            for sym, qty in holdings:
                try:
                    hist = yf.Ticker(sym).history(period="1mo")
                    if len(hist) < 14:
                        actions.append(f"⚠️ {sym}: Not enough historical data for RSI")
                        continue
                    delta = hist['Close'].diff()
                    gain = delta.clip(lower=0).rolling(14).mean()
                    loss = -delta.clip(upper=0).rolling(14).mean()
                    # Prevent division by zero
                    if loss.iloc[-1] == 0:
                        rsi = 100
                    else:
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs.iloc[-1]))
                    
                    price = hist['Close'].iloc[-1]
                    if rsi < 30:  # oversold → buy signal
                        trade_qty = round(50 / price, 4)
                        result = self.simulate_trade(sym, trade_qty, "buy", f"RSI={rsi:.1f} oversold signal")
                        actions.append(result)
                    elif rsi > 70:  # overbought → sell signal
                        sell_qty = round(qty * 0.1, 4)
                        if sell_qty > 0:
                            result = self.simulate_trade(sym, sell_qty, "sell", f"RSI={rsi:.1f} overbought signal")
                            actions.append(result)
                except Exception as e:
                    actions.append(f"⚠️ {sym}: {str(e)}")
            return "\n".join(actions) if actions else "No rebalancing needed. Portfolio is balanced."
        except Exception as e:
            return f"Rebalance error: {str(e)}"

    @staticmethod
    def get_trade_log(limit: int = 10) -> list:
        con = sqlite3.connect(PortfolioTool.DB)
        rows = con.execute(
            "SELECT ts, symbol, action, quantity, reason FROM trade_log ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        con.close()
        return [{"ts": r[0], "symbol": r[1], "action": r[2], "quantity": r[3], "reason": r[4]} for r in rows]


portfolio_tool = PortfolioTool()

# ── Specialist Agents ─────────────────────────────────────────────────────────
web_search_agent = Agent(
    name="Web Search Agent",
    role="Real-time market intelligence gathering",
    model=Groq(id=MODEL, parallel_tool_calls=False),
    tools=[DuckDuckGo()],
    instructions=[
        "Brevity is mandatory.",
        "1. Perform ONE comprehensive search for the topic.",
        "2. Do NOT attempt multiple searches at once.",
        "3. No greetings, no fluff, max 3 bullets.",
    ],

    storage=storage, add_history_to_messages=True, markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Quantitative financial analysis and portfolio management",
    model=Groq(id=MODEL, parallel_tool_calls=False),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True,
                      stock_fundamentals=True, company_news=True),
        portfolio_tool
    ],
    instructions=["Brevity mandatory. Use tools IMMEDIATELY for data. Bullet points only. No greetings."],
    storage=storage, add_history_to_messages=True, markdown=True,
)

sentiment_agent = Agent(
    name="Sentiment Agent",
    role="Psychological market mood analyzer",
    model=Groq(id=MODEL, parallel_tool_calls=False),
    instructions=["Brevity mandatory. Sentiment score 1-10 and 1-word mood only."],
    storage=storage, add_history_to_messages=True, markdown=True,
)


# ── Agent Team (Orchestrator) ─────────────────────────────────────────────────
agent_team = Agent(
    name="Lead Orchestrator",
    # Added direct tools to Orchestrator for speed
    tools=[
        delegate_to_finance_agent, 
        delegate_to_sentiment_agent, 
        delegate_to_web_search_agent,
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True),
        DuckDuckGo()
    ],
    model=Groq(id=MODEL, parallel_tool_calls=False),
    instructions=[
        "You are an Ultra-Lightweight Financial Orchestrator.",
        "1. Answer the user question DIRECTLY using your tools.",
        "2. IMPORTANT: If tools return JSON, summarize it in plain English. NEVER show raw JSON to the user.",
        "3. You can handle multiple symbols (e.g. AAPL, MSFT) in a single tool call.",
        "4. Consult the 'market_knowledge' base for theories.",
        "5. MANDATORY: Include: 'For educational purposes only. Not financial advice.'",
        "6. No greetings, no fluff, max 3-5 bullets.",
        "7. If you cannot find the answer after calling tools, say 'Error' and nothing else.",
    ],
    storage=storage, add_history_to_messages=True,
    knowledge_base=get_knowledge_base(),
    show_tool_calls=True, markdown=True,
)


# ── Agent Debate ──────────────────────────────────────────────────────────────
bull_agent = Agent(
    name="Bull Agent",
    role="Optimistic market analyst",
    model=Groq(id=MODEL, parallel_tool_calls=False),
    tools=[],
    instructions=[
        "You are an optimistic bull investor.",
        "Given a stock or market topic, argue strongly WHY it is a BUY.",
        "Use data, growth potential, and positive catalysts.",
        "Be concise — max 150 words.",
        "Do NOT call any tools or functions. Only respond with text.",
    ],
    markdown=True,
)

bear_agent = Agent(
    name="Bear Agent",
    role="Pessimistic risk analyst",
    model=Groq(id=MODEL, parallel_tool_calls=False),
    tools=[],
    instructions=[
        "You are a cautious bear investor.",
        "Given a stock or market topic, argue strongly WHY it is a SELL or AVOID.",
        "Use risks, overvaluation, and negative catalysts.",
        "Be concise — max 150 words.",
        "Do NOT call any tools or functions. Only respond with text.",
    ],
    markdown=True,
)

judge_agent = Agent(
    name="Judge Agent",
    role="Neutral arbitrator",
    model=Groq(id=MODEL, parallel_tool_calls=False),
    tools=[],
    instructions=[
        "You receive a Bull argument and a Bear argument about a stock.",
        "Weigh both sides objectively.",
        "Give a final VERDICT: BUY / HOLD / SELL with a confidence % and 2-line reasoning.",
        "Do NOT call any tools or functions. Only respond with text.",
    ],
    markdown=True,
)



def run_debate(topic: str) -> dict:
    """Run a 3-agent debate and return bull, bear, verdict."""
    try:
        bull_resp = run_agent_with_retry(bull_agent, f"Topic: {topic}")
        bull_content = bull_resp.content if bull_resp and bull_resp.content else "No bull argument generated."
        
        bear_resp = run_agent_with_retry(bear_agent, f"Topic: {topic}")
        bear_content = bear_resp.content if bear_resp and bear_resp.content else "No bear argument generated."
        
        verdict_resp = run_agent_with_retry(judge_agent, 
            f"Bull argument:\n{bull_content}\n\nBear argument:\n{bear_content}"
        )
        verdict_content = verdict_resp.content if verdict_resp and verdict_resp.content else "No verdict generated."
        
        return {
            "bull": bull_content,
            "bear": bear_content,
            "verdict": verdict_content,
        }
    except Exception as e:
        return {
            "bull": f"Error: {e}",
            "bear": f"Error: {e}",
            "verdict": f"Debate failed: {e}",
        }





# ── Scheduled Morning Brief ───────────────────────────────────────────────────
def run_morning_brief() -> str:
    """Generates a comprehensive market morning brief by running modular sub-tasks."""
    try:
        # Step 1: Global News
        news = run_agent_with_retry(web_search_agent, "Top 3 global market news for today. 3 bullets.")
        # Step 2: Portfolio Analysis
        portfolio = run_agent_with_retry(finance_agent, "Current portfolio value and RSI status of holdings.")
        # Step 3: Rebalance Check
        rebalance = run_agent_with_retry(finance_agent, "Execute portfolio rebalance if needed and summarize action.")
        
        brief = f"""
# ☕ Morning Market Brief
{datetime.now().strftime('%Y-%m-%d %H:%M')}

### 🌍 Global Headlines
{news.content if hasattr(news, 'content') else news}

### 📊 Portfolio & RSI Status
{portfolio.content if hasattr(portfolio, 'content') else portfolio}

### ⚖️ Rebalance Action
{rebalance.content if hasattr(rebalance, 'content') else rebalance}

*Disclaimer: For educational purposes only. Not financial advice.*
"""
        return brief
    except Exception as e:
        return f"Morning Brief Failed: {str(e)}"


def get_morning_briefs(limit: int = 5) -> list:
    try:
        con = sqlite3.connect(agent_storage)
        con.execute("""CREATE TABLE IF NOT EXISTS morning_briefs
                       (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, content TEXT)""")
        rows = con.execute(
            "SELECT ts, content FROM morning_briefs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        con.close()
        return [{"ts": r[0], "content": r[1]} for r in rows]
    except Exception:
        return []
