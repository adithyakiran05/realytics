import yfinance as yf
import pandas as pd

tickers = [
    "INFY.NS",
    "HINDUNILVR.NS",
    "TATAMOTORS.NS",
    "RELIANCE.NS",
    "ITC.NS"
]

def fetch_data():
    data = []
    failed = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)

            # ✅ Tier 1: Price data (MOST RELIABLE)
            hist = stock.history(period="1y")

            if hist.empty:
                print(f"{ticker} has no price data, skipping...")
                failed.append(ticker)
                continue

            avg_price = hist["Close"].mean()
            volatility = hist["Close"].std()

            # ⚠️ Tier 2: Try financials (optional)
            revenue = None
            ebitda = None

            try:
                financials = stock.financials

                if not financials.empty:
                    if "Total Revenue" in financials.index:
                        revenue = financials.loc["Total Revenue"].iloc[0]
                    if "EBITDA" in financials.index:
                        ebitda = financials.loc["EBITDA"].iloc[0]
            except:
                pass

            data.append({
                "Company": ticker,
                "Avg_Price": avg_price,
                "Volatility": volatility,
                "Revenue": revenue,
                "EBITDA": ebitda
            })

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            failed.append(ticker)

    df = pd.DataFrame(data)

    if not df.empty:
        df.to_csv("data/raw/market_data.csv", index=False)
        print("Data saved successfully")

    print("Failed tickers:", failed)


if __name__ == "__main__":
    fetch_data()