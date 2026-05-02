import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def train():
    df = pd.read_csv("data/processed/merged_data.csv")

    # Create target (dummy but logical)
    df["Future_EBITDA_Growth"] = df["Revenue_Growth"] * 1.2

    features = [
        "Revenue_Growth",
        "EBITDA_Margin",
        "ROCE",
        "Debt_Ratio",
        "Competition",
        "Public_Perception",
        "Industry_Risk"
    ]

    X = df[features]
    y = df["Future_EBITDA_Growth"]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    print("Model trained and saved")

if __name__ == "__main__":
    train()