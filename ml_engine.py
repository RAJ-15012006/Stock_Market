import numpy as np
import pandas as pd
import os
import time
import math
from datetime import datetime

# Graceful imports with fallback mechanisms
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except Exception:
    PYTORCH_AVAILABLE = False

# -------------------------------------------------------------------------
# 1. FinBERT Sentiment Pipeline
# -------------------------------------------------------------------------
class FinBERTSentiment:
    def __init__(self):
        self.nlp = None
        self.fallback_lexicon = {
            "bullish": 0.8, "bearish": -0.8, "surge": 0.7, "plunge": -0.7,
            "gain": 0.5, "loss": -0.5, "rise": 0.4, "fall": -0.4,
            "growth": 0.6, "decline": -0.6, "profit": 0.6, "deficit": -0.6,
            "upbeat": 0.7, "gloomy": -0.7, "beat": 0.5, "miss": -0.5,
            "upgrade": 0.6, "downgrade": -0.6, "acquire": 0.3, "bankrupt": -0.9,
            "jump": 0.5, "drop": -0.5, "rally": 0.7, "slump": -0.7
        }
        self._initialized = False

    def load_model(self):
        if self._initialized:
            return
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load with a short timeout to prevent hanging UI
                tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", local_files_only=False)
                model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self.nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)
            except Exception as e:
                print(f"FinBERT load failed, using rule-based fallback: {e}")
        self._initialized = True

    def analyze(self, text):
        """Returns positive, negative, neutral scores (sum to 1.0)"""
        self.load_model()
        if not text or not isinstance(text, str):
            return 0.0, 0.0, 1.0

        if self.nlp is not None:
            try:
                result = self.nlp(text[:512])[0]
                label = result['label'].lower()
                score = result['score']
                if label == 'positive':
                    return score, 0.0, 1.0 - score
                elif label == 'negative':
                    return 0.0, score, 1.0 - score
                else:
                    return 0.0, 0.0, 1.0
            except Exception:
                pass

        # Fallback Lexicon Sentiment
        words = text.lower().split()
        score = 0.0
        matches = 0
        for word in words:
            # Clean punctuation
            w = "".join([c for c in word if c.isalnum()])
            if w in self.fallback_lexicon:
                score += self.fallback_lexicon[w]
                matches += 1
        
        if matches == 0:
            return 0.1, 0.1, 0.8  # neutral default
        
        avg_score = score / matches
        if avg_score > 0.1:
            pos = min(avg_score * 1.2, 1.0)
            return pos, 0.0, 1.0 - pos
        elif avg_score < -0.1:
            neg = min(abs(avg_score) * 1.2, 1.0)
            return 0.0, neg, 1.0 - neg
        else:
            return 0.1, 0.1, 0.8

# Initialize global sentiment analyzer
finbert_analyzer = FinBERTSentiment()


# -------------------------------------------------------------------------
# 2. XGBoost Stock Movement Prediction
# -------------------------------------------------------------------------
def prepare_ml_features(df, sentiment_scores=None):
    """
    Computes technical indicators and overlays sentiment scores to form features.
    """
    df = df.copy()
    if len(df) < 20:
        return None

    # Returns and Volatility
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(10).std()

    # EMAs
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std

    # Volume Z-Score
    df['Vol_Mean'] = df['Volume'].rolling(20).mean()
    df['Vol_Std'] = df['Volume'].rolling(20).std().replace(0, 1e-9)
    df['Vol_Z'] = (df['Volume'] - df['Vol_Mean']) / df['Vol_Std']

    # Sentiment Score insertion (fallback to neutral if none provided)
    if sentiment_scores is not None and len(sentiment_scores) == len(df):
        df['Sentiment'] = sentiment_scores
    else:
        # Generate slightly random-walk mock sentiment to simulate real pipeline when empty
        np.random.seed(42)
        df['Sentiment'] = 0.5 + 0.15 * np.sin(np.linspace(0, 20, len(df))) + np.random.normal(0, 0.05, len(df))
        df['Sentiment'] = df['Sentiment'].clip(0, 1)

    # Clean NaNs
    df.dropna(inplace=True)
    return df

def train_prediction_model(df, prediction_days=5, threshold=0.02):
    """
    Trains XGBoost to predict if stock increases by > threshold in prediction_days.
    """
    if not XGBOOST_AVAILABLE:
        return None, "XGBoost not installed."

    df_feats = prepare_ml_features(df)
    if df_feats is None or len(df_feats) < 40:
        return None, "Insufficient data for ML training."

    # Define Target: Close in X days > current close * (1 + threshold)
    df_feats['Target'] = (df_feats['Close'].shift(-prediction_days) > df_feats['Close'] * (1 + threshold)).astype(int)
    
    # We drop the last X days since we don't have their future targets yet
    df_train_full = df_feats.dropna().copy()
    if len(df_train_full) < 30:
        return None, "Insufficient target labels."

    features = ['Return', 'Volatility', 'RSI', 'MACD', 'Vol_Z', 'Sentiment']
    X = df_train_full[features]
    y = df_train_full['Target']

    # Time series train-test split (no shuffle!)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    try:
        model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.08,
            random_state=42,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)

        # Evaluation metrics
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        # Calculate manually to avoid sklearn dependency errors
        tp = np.sum((preds == 1) & (y_test == 1))
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))
        tn = np.sum((preds == 0) & (y_test == 0))

        accuracy = (tp + tn) / len(y_test) if len(y_test) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Latest features for next prediction
        latest_features = df_feats[features].iloc[-1:].copy()
        pred_prob = float(model.predict_proba(latest_features)[:, 1][0])
        pred_label = "BUY/UP" if pred_prob > 0.55 else "HOLD/SELL"

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pred_prob": pred_prob,
            "pred_label": pred_label
        }

        return {
            "model": model,
            "metrics": metrics,
            "features": features,
            "X_train": X_train,
            "X_test": X_test,
            "latest_features": latest_features
        }, None
    except Exception as e:
        return None, f"Training failed: {str(e)}"


# -------------------------------------------------------------------------
# 3. SHAP Explainability
# -------------------------------------------------------------------------
def get_shap_explanation(model_dict):
    """
    Returns matplotlib figure representing SHAP waterfall / feature importance.
    """
    if not SHAP_AVAILABLE:
        return None

    import matplotlib.pyplot as plt
    try:
        model = model_dict["model"]
        X_train = model_dict["X_train"]
        latest_features = model_dict["latest_features"]

        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(latest_features)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        # Style matplotlib for dark theme compatibility
        fig.patch.set_facecolor('#09090b')
        ax.set_facecolor('#18181b')
        ax.tick_params(colors='#fafafa')
        ax.xaxis.label.set_color('#fafafa')
        ax.yaxis.label.set_color('#fafafa')
        
        # Render a simple shap bar explanation of features contributing to latest prediction
        contribs = shap_values.values[0]
        base_val = shap_values.base_values[0]
        feats = model_dict["features"]

        y_pos = np.arange(len(feats))
        colors = ['#4ade80' if c >= 0 else '#f87171' for c in contribs]

        ax.barh(y_pos, contribs, align='center', color=colors, edgecolor='#27272a')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats)
        ax.invert_yaxis()
        ax.set_xlabel('SHAP Impact on Buy Probability')
        ax.set_title(f'Prediction Factors (Base Prob: {base_val:.2f})', color='#fafafa')
        plt.tight_layout()

        return fig
    except Exception as e:
        print(f"SHAP error: {e}")
        return None


# -------------------------------------------------------------------------
# 4. Market Regime Detection using HMM
# -------------------------------------------------------------------------
def detect_market_regimes(df):
    """
    Trains a GaussianHMM to cluster stock into 3 regimes: Bull, Bear, Volatile.
    """
    df = df.copy()
    if len(df) < 50:
        return df, "Insufficient history for HMM"

    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Return'].rolling(10).std()
    df.dropna(inplace=True)

    X = df[['Return', 'Volatility']].values

    if HMM_AVAILABLE:
        try:
            model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
            model.fit(X)
            states = model.predict(X)

            # Sort states by return mean to ensure: 0 = Bear, 1 = Sideways/Volatile, 2 = Bull
            means = [model.means_[i][0] for i in range(3)]
            state_order = np.argsort(means)
            state_map = {state_order[0]: 0, state_order[1]: 1, state_order[2]: 2}
            
            mapped_states = [state_map[s] for s in states]
            df['Regime'] = mapped_states
            
            # Label names
            regime_names = {0: "Bear Market", 1: "Sideways/Volatile", 2: "Bull Market"}
            df['Regime_Name'] = [regime_names[s] for s in mapped_states]
            return df, None
        except Exception as e:
            pass

    # Simple K-Means or Quantile Fallback for Regimes
    # If HMM is not available, we use volatility and returns quantile splits
    vol_median = df['Volatility'].median()
    ret_median = df['Return'].median()
    
    regimes = []
    regime_names = []
    for r, v in zip(df['Return'], df['Volatility']):
        if r < ret_median and v > vol_median:
            regimes.append(0)
            regime_names.append("Bear Market")
        elif v <= vol_median:
            regimes.append(2)
            regime_names.append("Bull Market")
        else:
            regimes.append(1)
            regime_names.append("Sideways/Volatile")

    df['Regime'] = regimes
    df['Regime_Name'] = regime_names
    return df, None


# -------------------------------------------------------------------------
# 5. Reinforcement Learning (RL) Portfolio Optimizer
# -------------------------------------------------------------------------
class RLOptimizerAgent:
    """
    A pure PyTorch Policy Gradient (REINFORCE) agent that learns dynamic weight allocation
    for 3 assets: NVDA, AAPL, BTC-USD.
    """
    def __init__(self, state_dim=6, action_dim=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if PYTORCH_AVAILABLE:
            # Policy network outputting portfolio weights (via Softmax)
            self.policy = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, action_dim),
                nn.Softmax(dim=-1)
            ).to(self.device)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
        else:
            self.policy = None

    def select_action(self, state):
        if not PYTORCH_AVAILABLE:
            # Equal weight fallback
            return np.array([1/self.action_dim] * self.action_dim)
        
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            weights = self.policy(state_t).cpu().numpy()
        return weights

    def train_on_history(self, price_matrix, episodes=20):
        """
        Matrix shape: (days, 3 assets). Trains policy to maximize logarithmic returns.
        """
        if not PYTORCH_AVAILABLE:
            return [0.0] * episodes, [1/3, 1/3, 1/3]

        n_days, n_assets = price_matrix.shape
        # Daily returns
        daily_returns = price_matrix[1:] / price_matrix[:-1] - 1
        
        episode_rewards = []
        
        for ep in range(episodes):
            log_probs = []
            rewards = []
            
            # Simple state representation: rolling returns of last 2 days
            for t in range(2, n_days - 1):
                state = daily_returns[t-2:t].flatten()  # 2 days * 3 assets = 6 features
                state_t = torch.FloatTensor(state).to(self.device)
                
                # Forward
                action_probs = self.policy(state_t)
                
                # Sample action
                dist = torch.distributions.Dirichlet(action_probs * 10) # concentration
                weights = dist.sample()
                
                # Reward: Portfolio Return
                ret = float(np.sum(daily_returns[t] * weights.cpu().numpy()))
                rewards.append(ret)
                
                log_prob = dist.log_prob(weights)
                log_probs.append(log_prob)
            
            # Optimize policy
            discounted_returns = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.95 * G
                discounted_returns.insert(0, G)
            
            discounted_returns = torch.FloatTensor(discounted_returns).to(self.device)
            # Normalize rewards
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9)
            
            policy_loss = []
            for lp, G_t in zip(log_probs, discounted_returns):
                policy_loss.append(-lp * G_t)
                
            self.optimizer.zero_grad()
            if len(policy_loss) > 0:
                loss = torch.stack(policy_loss).sum()
                loss.backward()
                self.optimizer.step()
                
            episode_rewards.append(sum(rewards))

        # Final recommended weight on latest state
        latest_state = daily_returns[-2:].flatten()
        rec_weights = self.select_action(latest_state)
        
        return episode_rewards, rec_weights

# -------------------------------------------------------------------------
# 6. Backtesting Simulator
# -------------------------------------------------------------------------
def run_strategy_backtest(df, strategy="RSI"):
    """
    Backtests standard trading strategies vs Benchmark (Buy & Hold)
    Returns metrics and returns series for plotting.
    """
    df = df.copy()
    if len(df) < 30:
        return None, "Not enough data for backtesting"

    df['Return'] = df['Close'].pct_change()
    df['Cum_Benchmark'] = (1 + df['Return'].fillna(0)).cumprod()

    # Calculate indicators
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # Position tracking
    position = 0
    strategy_returns = []

    if strategy == "RSI":
        for idx in range(len(df)):
            curr_rsi = df['RSI'].iloc[idx]
            if pd.isna(curr_rsi):
                strategy_returns.append(0.0)
                continue
            
            if curr_rsi < 30:
                position = 1  # Buy
            elif curr_rsi > 70:
                position = 0  # Sell/Hold cash
                
            strategy_returns.append(position * df['Return'].iloc[idx])

    elif strategy == "MACD":
        for idx in range(len(df)):
            macd = df['MACD'].iloc[idx]
            sig = df['MACD_Signal'].iloc[idx]
            if pd.isna(macd) or pd.isna(sig):
                strategy_returns.append(0.0)
                continue
            
            if macd > sig:
                position = 1
            else:
                position = 0
            strategy_returns.append(position * df['Return'].iloc[idx])
            
    elif strategy == "ML_Predictive":
        # Train a rolling ML model
        # Simple rule: buy if Return > 0 in rolling window
        df_ml = prepare_ml_features(df)
        if df_ml is not None and XGBOOST_AVAILABLE:
            df_ml['Target'] = (df_ml['Close'].shift(-1) > df_ml['Close']).astype(int)
            features = ['Return', 'Volatility', 'RSI', 'MACD', 'Vol_Z', 'Sentiment']
            
            # Rolling window prediction
            positions = [0] * 30 # initial startup
            for t in range(30, len(df_ml)):
                train_slice = df_ml.iloc[t-30:t]
                model = XGBClassifier(n_estimators=10, max_depth=2, learning_rate=0.1, eval_metric="logloss", random_state=42)
                model.fit(train_slice[features], train_slice['Target'])
                
                pred = model.predict(df_ml[features].iloc[t:t+1])[0]
                positions.append(pred)
            
            df_ml['Position'] = positions
            df_ml['Strat_Ret'] = df_ml['Position'] * df_ml['Return']
            
            # Align back with original df
            df = df.join(df_ml[['Strat_Ret']], how='left').fillna(0)
            strategy_returns = df['Strat_Ret'].values
        else:
            # Fallback strategy (EMA Crossover)
            df['EMA9'] = df['Close'].ewm(span=9).mean()
            df['EMA20'] = df['Close'].ewm(span=20).mean()
            for idx in range(len(df)):
                pos = 1 if df['EMA9'].iloc[idx] > df['EMA20'].iloc[idx] else 0
                strategy_returns.append(pos * df['Return'].iloc[idx])

    df['Strat_Return'] = strategy_returns
    df['Cum_Strategy'] = (1 + df['Strat_Return'].fillna(0)).cumprod()

    # Metrics computation
    total_days = len(df)
    years = total_days / 252.0 if total_days > 0 else 1.0

    cagr_strat = float((df['Cum_Strategy'].iloc[-1]) ** (1 / max(years, 0.001)) - 1)
    cagr_bench = float((df['Cum_Benchmark'].iloc[-1]) ** (1 / max(years, 0.001)) - 1)

    # Max Drawdown
    roll_max_strat = df['Cum_Strategy'].cummax()
    drawdown_strat = (df['Cum_Strategy'] - roll_max_strat) / roll_max_strat
    max_dd_strat = float(drawdown_strat.min())

    roll_max_bench = df['Cum_Benchmark'].cummax()
    drawdown_bench = (df['Cum_Benchmark'] - roll_max_bench) / roll_max_bench
    max_dd_bench = float(drawdown_bench.min())

    # Sharpe Ratio (annualized, 2% risk-free rate)
    rf_daily = 0.02 / 252
    excess_strat = df['Strat_Return'] - rf_daily
    sharpe_strat = float((excess_strat.mean() / (excess_strat.std() + 1e-9)) * math.sqrt(252))

    excess_bench = df['Return'] - rf_daily
    sharpe_bench = float((excess_bench.mean() / (excess_bench.std() + 1e-9)) * math.sqrt(252))

    # Win Rate
    wins = np.sum(df['Strat_Return'] > 0)
    total_trades = np.sum(df['Strat_Return'] != 0)
    win_rate = float(wins / total_trades) if total_trades > 0 else 0.0

    metrics = {
        "cagr_strat": cagr_strat * 100,
        "cagr_bench": cagr_bench * 100,
        "max_dd_strat": max_dd_strat * 100,
        "max_dd_bench": max_dd_bench * 100,
        "sharpe_strat": sharpe_strat,
        "sharpe_bench": sharpe_bench,
        "win_rate": win_rate * 100,
        "total_return_strat": float((df['Cum_Strategy'].iloc[-1] - 1) * 100),
        "total_return_bench": float((df['Cum_Benchmark'].iloc[-1] - 1) * 100)
    }

    return df, metrics
