import yfinance as yf
import json

def get_nvda_analysis():
    nvda = yf.Ticker("NVDA")
    
    # 1. Price
    price = nvda.fast_info.last_price
    
    # 2. Fundamentals
    info = nvda.info
    fundamentals = {
        "Market Cap": info.get("marketCap"),
        "Forward P/E": info.get("forwardPE"),
        "Revenue Growth": info.get("revenueGrowth"),
        "Profit Margin": info.get("profitMargins"),
    }
    
    # 3. Recommendations
    recs = nvda.recommendations
    latest_recs = recs.tail(5).to_dict(orient="records") if recs is not None and not recs.empty else "No recent recommendations"

    print(json.dumps({
        "price": price,
        "fundamentals": fundamentals,
        "recommendations": latest_recs
    }, indent=2))

if __name__ == "__main__":
    get_nvda_analysis()
