from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from financial_agent import agent_team
import yfinance as yf
from typing import List, Optional

app = FastAPI(title="Agentic Finance API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    prompt: str

class StockRequest(BaseModel):
    symbol: str

@app.post("/chat")
async def chat(query: Query):
    try:
        response = agent_team.run(query.prompt)
        return {"content": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stock/{symbol}")
async def get_stock(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1mo")
        
        # Format for frontend
        chart_data = [{"time": str(date.date()), "value": price} for date, price in hist['Close'].items()]
        
        return {
            "symbol": symbol,
            "price": info.get("currentPrice"),
            "change": info.get("revenueGrowth", 0) * 100,
            "marketCap": info.get("marketCap"),
            "history": chart_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
