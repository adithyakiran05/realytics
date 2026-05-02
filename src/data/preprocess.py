import pandas as pd

def merge_data():
    market = pd.read_csv("data/raw/market_data.csv")
    qualitative = pd.read_csv("data/external/qualitative_data.csv")

    df = pd.merge(market, qualitative, on="Company")

    # Dummy ML target (for now)
    df["Future_EBITDA_Growth"] = df["Revenue_Growth"] * 1.2

    df.to_csv("data/processed/merged_data.csv", index=False)
    print("Merged data created")

if __name__ == "__main__":
    merge_data()