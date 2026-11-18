from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.db.sqlite import SqliteDb
from agno.tools import Toolkit
from dotenv import load_dotenv
import yfinance as yf
import json, os, sqlite3, time
from datetime import datetime
import functools


# Load .env FIRST so API keys are available before anything else
load_dotenv()

# Automatically sync Streamlit Cloud Secrets into environment variables if available
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for k in st.secrets:
            if k not in os.environ:
                os.environ[k] = str(st.secrets[k])
except Exception:
    pass


# Morse observability initialization
try:
    import morse_ai
    morse_ai.init()
    MORSE_ENABLED = True
    print("[Morse] SDK initialized")
except Exception as e:
    morse_ai = None
    MORSE_ENABLED = False
    print(f"[Morse] Observability unavailable: {e}")

try:
    from agno.models.google import Gemini
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    Gemini = None







def _install_morse_groq_instrumentation():
    """Capture Groq model usage and calculate cost in Morse."""

    if not MORSE_ENABLED:
        return

    original_invoke = Groq.invoke

    if getattr(original_invoke, "_morse_instrumented", False):
        return

    @functools.wraps(original_invoke)
    def morse_instrumented_invoke(self, *args, **kwargs):
        model_id = getattr(self, "id", "unknown")

        try:
            with morse_ai.span(
                f"Groq LLM Request: {model_id}",
                span_type="llm",
                input_data={
                    "provider": "Groq",
                    "model": model_id,
                },
            ) as span:

                response = original_invoke(self, *args, **kwargs)
                usage = getattr(response, "response_usage", None)

                if usage is not None:
                    span_data = getattr(span, "_span", None)

                    if span_data is not None:
                        input_tokens = getattr(
                            usage, "input_tokens", None
                        )
                        output_tokens = getattr(
                            usage, "output_tokens", None
                        )
                        total_tokens = getattr(
                            usage, "total_tokens", None
                        )
                        reasoning_tokens = getattr(
                            usage, "reasoning_tokens", None
                        )

                        span_data.model = model_id
                        span_data.input_tokens = input_tokens
                        span_data.output_tokens = output_tokens
                        span_data.total_tokens = total_tokens
                        span_data.reasoning_tokens = reasoning_tokens

                        # Groq published on-demand pricing (USD per 1M tokens).
                        pricing = {
                            "openai/gpt-oss-120b": {
                                "input": 0.15,
                                "cached_input": 0.075,
                                "output": 0.60,
                            },
                            "openai/gpt-oss-20b": {
                                "input": 0.075,
                                "cached_input": 0.037,
                                "output": 0.30,
                            },
                        }

                        rates = pricing.get(model_id)

                        cached_tokens = (
                            getattr(usage, "cache_read_tokens", 0) or 0
                        )

                        if (
                            rates is not None
                            and input_tokens is not None
                            and output_tokens is not None
                        ):
                            cached_tokens = min(
                                cached_tokens, input_tokens
                            )
                            regular_input = input_tokens - cached_tokens

                            cost_usd = (
                                regular_input * rates["input"]
                                + cached_tokens * rates["cached_input"]
                                + output_tokens * rates["output"]
                            ) / 1_000_000

                            span_data.cost_usd = cost_usd

                            print(
                                f"[Morse] {model_id} | "
                                f"Input: {input_tokens} | "
                                f"Output: {output_tokens} | "
                                f"Cost: ${cost_usd:.10f}"
                            )
                        else:
                            print(
                                f"[Morse] Usage captured for {model_id}; "
                                "cost unavailable for this model or usage."
                            )

                return response

        except Exception as exc:
            print(f"[Morse] Model instrumentation error: {exc}")
            raise

    morse_instrumented_invoke._morse_instrumented = True
    Groq.invoke = morse_instrumented_invoke

    print("[Morse] Groq native usage and cost instrumentation installed")


_install_morse_groq_instrumentation()


def _install_morse_gemini_instrumentation():
    """Capture Gemini model usage in Morse native LLM span fields."""

    if not MORSE_ENABLED or not _GEMINI_AVAILABLE:
        return

    original_invoke = Gemini.invoke

    if getattr(original_invoke, "_morse_instrumented", False):
        return

    @functools.wraps(original_invoke)
    def morse_instrumented_invoke(self, *args, **kwargs):
        model_id = getattr(self, "id", "unknown")

        try:
            with morse_ai.span(
                f"Gemini LLM Request: {model_id}",
                span_type="llm",
                input_data={
                    "provider": "Google Gemini",
                    "model": model_id,
                },
            ) as span:

                response = original_invoke(self, *args, **kwargs)
                usage = getattr(response, "response_usage", None)

                if usage is not None:
                    span_data = getattr(span, "_span", None)

                    if span_data is not None:
                        input_tokens = getattr(
                            usage, "input_tokens", None
                        )
                        output_tokens = getattr(
                            usage, "output_tokens", None
                        )
                        total_tokens = getattr(
                            usage, "total_tokens", None
                        )

                        span_data.model = model_id
                        span_data.input_tokens = input_tokens
                        span_data.output_tokens = output_tokens
                        span_data.total_tokens = total_tokens
                        span_data.reasoning_tokens = getattr(
                            usage, "reasoning_tokens", None
                        )

                        # Do not invent Gemini pricing.
                        provider_cost = getattr(usage, "cost", None)

                        if provider_cost is not None:
                            span_data.cost_usd = provider_cost

                        print(
                            f"[Morse] Gemini {model_id} | "
                            f"Input: {input_tokens} | "
                            f"Output: {output_tokens} | "
                            f"Total: {total_tokens}"
                        )

                return response

        except Exception as exc:
            print(f"[Morse] Gemini instrumentation error: {exc}")
            raise

    morse_instrumented_invoke._morse_instrumented = True
    Gemini.invoke = morse_instrumented_invoke

    print("[Morse] Gemini native usage instrumentation installed")


_install_morse_gemini_instrumentation()













def _detect_best_groq_model() -> tuple[str, list[str]]:
    """Detect active text models on Groq for this API key dynamically."""
    fallback_candidates = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "allam-2-7b",
    ]
    try:
        from groq import Groq as GroqClient
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return fallback_candidates[0], fallback_candidates
        client = GroqClient(api_key=api_key)
        all_ids = [m.id for m in client.models.list().data if getattr(m, "active", True)]
        chat_ids = [
            m for m in all_ids 
            if not any(x in m for x in ["whisper", "prompt-guard", "orpheus", "safeguard", "vision"])
        ]
        ordered = [m for m in fallback_candidates if m in chat_ids]
        for m in chat_ids:
            if m not in ordered:
                ordered.append(m)
        if ordered:
            return ordered[0], ordered
    except Exception as e:
        print(f"Groq model discovery notice: {e}")
    return fallback_candidates[0], fallback_candidates

_PRIMARY_MODEL, _GROQ_FALLBACK_MODELS = _detect_best_groq_model()
MODEL = _PRIMARY_MODEL

# ── Valid Gemini model IDs for fallback ────────────────────────────────────────
_GEMINI_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
]
agent_storage = "agent_sessions.db"
storage = SqliteDb(db_file=agent_storage, session_table="financial_assistant")


def _extract_content(resp) -> str:
    """Safely extract text content from an agno RunResponse.
    In newer agno versions resp.content may be a list of TextContent objects
    or a plain string. This handles both cases."""
    if resp is None:
        return ""
    # Plain string (older agno / FallbackResponse)
    if isinstance(resp.content, str):
        return resp.content
    # List of content blocks (newer agno)
    if isinstance(resp.content, list):
        parts = []
        for block in resp.content:
            if isinstance(block, str):
                parts.append(block)
            elif hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "content"):
                parts.append(str(block.content))
            else:
                parts.append(str(block))
        return "\n".join(parts).strip()
    # Fallback: coerce to string
    return str(resp.content) if resp.content else ""


def _is_error_response(resp) -> bool:
    """Check if agent output is an API error JSON string instead of actual content."""
    if not resp or not hasattr(resp, 'content') or not resp.content:
        return False
    content_str = _extract_content(resp).strip()
    return (
        content_str.startswith('{"error"')
        or '"error":' in content_str
        or '"invalid_request_error"' in content_str
        or '"model_not_found"' in content_str
        or '"NOT_FOUND"' in content_str
        or "no longer available" in content_str
        or "does not exist" in content_str
        or "do not have access" in content_str
    )


def _is_model_not_found_error(err_str: str) -> bool:
    """Returns True if the error string indicates an invalid/unavailable model."""
    return (
        "model_not_found" in err_str
        or "does not exist" in err_str
        or "do not have access to it" in err_str
        or "model not found" in err_str.lower()
        or "no longer available" in err_str
    )



def _run_agent_with_retry_untraced(agent, prompt, max_retries=3):
    """
    Run an Agno agent with:
    1. Original Groq model
    2. Gemini fallback
    3. Alternative Groq models

    Preserves the original model after execution.
    Captures provider-level telemetry through Morse instrumentation.
    """

    original_model = agent.model
    last_err_msg = ""

    def classify_error(message):
        """Classify common LLM provider errors."""

        msg = str(message).lower()

        if any(term in msg for term in [
            "model_not_found",
            "model not found",
            "no longer available",
            "does not exist",
            "not_found",
            "do not have access",
        ]):
            return "model_not_found"

        if any(term in msg for term in [
            "429",
            "rate_limit",
            "resource_exhausted",
            "too many requests",
        ]):
            return "rate_limit"

        if any(term in msg for term in [
            "503",
            "502",
            "504",
            "unavailable",
            "service unavailable",
        ]):
            return "provider_unavailable"

        if any(term in msg for term in [
            "401",
            "403",
            "unauthorized",
            "permission_denied",
            "invalid api key",
        ]):
            return "authentication"

        return "other"

    try:

        # =====================================================
        # PRIMARY PROVIDER: ORIGINAL GROQ MODEL
        # =====================================================

        for i in range(max_retries):

            try:
                resp = agent.run(prompt, stream=False)
                extracted = _extract_content(resp)

                if resp and extracted and not _is_error_response(resp):
                    resp.content = extracted
                    return resp

                last_err_msg = extracted or "Empty primary model response."
                error_type = classify_error(last_err_msg)

            except Exception as exc:
                last_err_msg = str(exc)
                error_type = classify_error(last_err_msg)

            print(
                f"[Primary] Attempt {i + 1}/{max_retries} failed "
                f"({error_type}): {last_err_msg[:180]}"
            )

            # Invalid model, provider outage or invalid credentials:
            # stop retrying the same primary model.
            if error_type in (
                "model_not_found",
                "provider_unavailable",
                "authentication",
            ):
                break

            # Controlled retry for rate limits and other errors.
            if i < max_retries - 1:
                time.sleep(min(2 ** i, 8))

        # =====================================================
        # FALLBACK 1: GOOGLE GEMINI
        # =====================================================

        if _GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):

            for g_model_id in _GEMINI_FALLBACK_MODELS:

                try:
                    agent.model = Gemini(
                        id=g_model_id,
                        api_key=os.getenv("GOOGLE_API_KEY"),
                    )

                    resp = agent.run(prompt, stream=False)
                    extracted = _extract_content(resp)

                    if resp and extracted and not _is_error_response(resp):
                        resp.content = extracted

                        print(
                            f"[Fallback] Gemini succeeded: {g_model_id}"
                        )

                        return resp

                    last_err_msg = extracted or "Empty Gemini response."
                    error_type = classify_error(last_err_msg)

                except Exception as exc:
                    last_err_msg = str(exc)
                    error_type = classify_error(last_err_msg)

                print(
                    f"[Fallback] Gemini {g_model_id} failed "
                    f"({error_type}): {last_err_msg[:180]}"
                )

                # Try another Gemini model only for an unavailable model ID.
                if error_type == "model_not_found":
                    continue

                # Provider-wide errors: stop cycling through Gemini models.
                if error_type in (
                    "provider_unavailable",
                    "rate_limit",
                    "authentication",
                ):
                    break

        # =====================================================
        # FALLBACK 2: ALTERNATIVE GROQ MODELS
        # =====================================================

        if os.getenv("GROQ_API_KEY"):

            original_model_id = getattr(original_model, "id", None)

            for alt_id in _GROQ_FALLBACK_MODELS:

                # Do not execute the same model twice.
                if alt_id == original_model_id:
                    continue

                try:
                    agent.model = Groq(
                        id=alt_id,
                        api_key=os.getenv("GROQ_API_KEY", ""),
                    )

                    resp = agent.run(prompt, stream=False)
                    extracted = _extract_content(resp)

                    if resp and extracted and not _is_error_response(resp):
                        resp.content = extracted

                        print(
                            f"[Fallback] Alternative Groq succeeded: {alt_id}"
                        )

                        return resp

                    last_err_msg = extracted or "Empty Groq response."
                    error_type = classify_error(last_err_msg)

                except Exception as exc:
                    last_err_msg = str(exc)
                    error_type = classify_error(last_err_msg)

                print(
                    f"[Fallback] Groq {alt_id} failed "
                    f"({error_type}): {last_err_msg[:180]}"
                )

                if error_type == "model_not_found":
                    continue

                if error_type in (
                    "provider_unavailable",
                    "rate_limit",
                    "authentication",
                ):
                    break

        # =====================================================
        # ALL PROVIDERS FAILED
        # =====================================================

        class FallbackResponse:
            def __init__(self, content):
                self.content = content

        return FallbackResponse(
            "⚠️ **AI Model Error**: Could not complete the query "
            "with any available LLM provider.\n\n"
            f"*Details:* `{last_err_msg[:300]}`\n\n"
            "💡 *Tip: Check your GROQ_API_KEY and GOOGLE_API_KEY "
            "in Streamlit Secrets or your .env file.*"
        )

    finally:
        # Always restore the originally configured agent model.
        agent.model = original_model


def run_agent_with_retry(agent, prompt, max_retries=3):
    """
    Execute an agent with Morse observability.

    - Inside an existing trace: create a child agent span.
    - Outside a trace: create a standalone agent trace.
    - Preserve existing retry and fallback behavior.
    """
    agent_name = getattr(agent, "name", None) or "Unknown Agent"

    def execute():
        return _run_agent_with_retry_untraced(
            agent, prompt, max_retries=max_retries
        )

    def is_successful(response):
        content = _extract_content(response)
        return (
            bool(content.strip())
            and not content.lstrip().startswith(
                "⚠️ **AI Model Error**"
            )
        )

    # Morse disabled: preserve original execution.
    if morse_ai is None:
        return execute()

    from morse_ai.tracing import get_active_trace

    active_trace = get_active_trace()

    # CASE 1: Already inside the Lead Orchestrator trace.
    if active_trace is not None:
        with morse_ai.span(
            agent_name,
            span_type="agent",
            input_data={
                "agent": agent_name,
                "prompt": str(prompt)[:1000],
            },
        ) as span:
            response = execute()
            success = is_successful(response)

            span.set_metadata("agent_name", agent_name)
            span.set_metadata("success", success)
            span.set_output({
                "response_preview": _extract_content(response)[:2000]
            })

            if not success:
                span.mark_failed("Agent returned an error response")

            return response

    # CASE 2: No active trace. Create a standalone trace.
    with morse_ai.run(agent_name) as trace:
        response = execute()
        success = is_successful(response)

        trace.set_outcome(
            success=success,
            outcome="completed" if success else "model_error",
        )

        return response

def delegate_to_finance_agent(task: str, symbol: str = "None") -> str:
    """Get financial data, price, fundamentals for a stock symbol."""
    try:
        return _extract_content(run_agent_with_retry(finance_agent, f"Task: {task}. Symbol: {symbol}"))
    except Exception as e:
        return f"Finance error: {str(e)}"


def delegate_to_web_search_agent(task: str, symbol: str = "None") -> str:
    """Search latest news and market updates for a topic or symbol."""
    try:
        return _extract_content(run_agent_with_retry(web_search_agent, f"Task: {task}. Symbol: {symbol}"))
    except Exception as e:
        return f"Search error: {str(e)}"


def delegate_to_sentiment_agent(task: str, symbol: str = "None") -> str:
    """Get market sentiment score and mood for a symbol."""
    try:
        return _extract_content(run_agent_with_retry(sentiment_agent, f"Task: {task}. Symbol: {symbol}"))
    except Exception as e:
        return f"Sentiment error: {str(e)}"


def set_model(model_id: str):
    global MODEL
    MODEL = model_id
    try:
        if "gemini" in MODEL.lower():
            if not _GEMINI_AVAILABLE:
                return "Gemini unavailable: google-genai not installed."
            new_model = Gemini(id=MODEL, api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            new_model = Groq(id=MODEL, api_key=os.getenv("GROQ_API_KEY", ""))
        for ag in [web_search_agent, finance_agent, sentiment_agent,
                   agent_team, bull_agent, bear_agent, judge_agent]:
            ag.model = new_model
        return f"Model switched to {model_id}"
    except Exception as e:
        return f"Failed to switch model: {e}"


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

# ── Custom Quant & RAG Tools for Agents ──────────────────────────────────────
def query_stock_market_guide(query: str) -> str:
    """Queries the local 150-page Stock Market Master Guide PDF for textbook/educational answers."""
    try:
        from rag_engine import rag_assistant
        results = rag_assistant.query(query, k=3)
        if not results:
            return "No matching information found in the Guide."
        response = []
        for r in results:
            response.append(f"Content: {r['text']}\nSource: {r['source']} (Page {r['page']})")
        return "\n\n".join(response)
    except Exception as e:
        return f"Error querying guide: {e}"

def get_ml_movement_prediction(symbol: str) -> str:
    """Gets the machine learning movement prediction (XGBoost, SHAP factors, and HMM regimes) for a stock symbol."""
    try:
        import yfinance as yf
        from ml_engine import train_prediction_model, detect_market_regimes
        hist = yf.Ticker(symbol).history(period="1y")
        if hist.empty:
            return f"No stock data found for {symbol}."
        res, err = train_prediction_model(hist)
        if err:
            return f"ML Prediction error: {err}"
        
        reg_df, reg_err = detect_market_regimes(hist)
        regime = reg_df['Regime_Name'].iloc[-1] if reg_err is None else "Unknown"
        
        metrics = res["metrics"]
        return f"Prediction: {metrics['pred_label']}\nProbability: {metrics['pred_prob']*100:.1f}%\n" \
               f"Model Performance (XGBoost): Accuracy={metrics['accuracy']*100:.1f}%, F1={metrics['f1']*100:.1f}%\n" \
               f"Market Regime (HMM): {regime}"
    except Exception as e:
        return f"Error running ML: {e}"

# ── Agents ────────────────────────────────────────────────────────────────────
_STRICT = [
    "Reply in MAX 5 bullet points.",
    "Each bullet: one short sentence.",
    "NO greetings, NO disclaimers, NO filler.",
    "Use tools once. Do not repeat tool calls.",
]

# Helper: build a Groq model instance with explicit API key
def _groq(model_id: str = MODEL):
    return Groq(id=model_id, api_key=os.getenv("GROQ_API_KEY", ""))

web_search_agent = Agent(
    name="Web Search Agent",
    role="Real-time market news",
    model=_groq(),
    tools=[DuckDuckGoTools()],
    instructions=_STRICT + ["Search once. Return top 3 news bullets only."],
    db=storage, add_history_to_context=False, markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Stock data and portfolio analysis",
    model=_groq(),
    tools=[
        YFinanceTools(enable_stock_price=True, enable_analyst_recommendations=True,
                      enable_stock_fundamentals=True, enable_company_news=True),
        portfolio_tool,
        get_ml_movement_prediction,
    ],
    instructions=_STRICT + ["Fetch data with tools. Show price, PE, recommendation, RSI in bullets."],
    db=storage, add_history_to_context=True, markdown=True,
)

sentiment_agent = Agent(
    name="Sentiment Agent",
    role="Market mood",
    model=_groq(),
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
        query_stock_market_guide,
        get_ml_movement_prediction,
    ],
    model=_groq(),
    instructions=[
        "You are a financial AI. Answer the user question using tools.",
        "When asked textbook financial questions or definitions, use query_stock_market_guide tool to cite sources/page numbers.",
        "When asked to predict or forecast movements or evaluate machine learning, use get_ml_movement_prediction.",
        "When giving stock buy/sell/hold recommendations, ALWAYS structure your response as:",
        "  - Signal: [BUY / HOLD / SELL]",
        "  - Confidence Score: [0-100%]",
        "  - Expected Return: [Expected percentage return]",
        "  - Risk Level: [Low / Moderate / High]",
        "  - Reasoning: [Brief explanation using LLM + SHAP factors]",
        "Otherwise, use MAX 5 bullet points where each bullet is 1 short sentence.",
        "NEVER show raw JSON. Summarize numbers in plain English.",
        "Call each tool ONCE. Do not loop.",
        "End every reply with: '⚠️ Educational only. Not financial advice.'",
    ],
    db=storage, add_history_to_context=True, markdown=True,
)

bull_agent = Agent(
    name="Bull Agent", role="Optimistic analyst",
    model=_groq(), tools=[],
    instructions=["Argue BUY in exactly 4 bullets. Max 120 words total. No tools."],
    markdown=True,
)

bear_agent = Agent(
    name="Bear Agent", role="Pessimistic analyst",
    model=_groq(), tools=[],
    instructions=["Argue SELL in exactly 4 bullets. Max 120 words total. No tools."],
    markdown=True,
)

judge_agent = Agent(
    name="Judge Agent", role="Neutral arbitrator",
    model=_groq(), tools=[],
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
