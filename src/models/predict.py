import pandas as pd
import joblib

def predict():
    df = pd.read_csv("data/processed/merged_data.csv")
    model = joblib.load("model.pkl")

    features = [
        "Revenue_Growth",
        "EBITDA_Margin",
        "ROCE",
        "Debt_Ratio",
        "Competition",
        "Public_Perception",
        "Industry_Risk"
    ]

    df["Predicted_Growth"] = model.predict(df[features])

    df.to_csv("data/processed/predictions.csv", index=False)
    print("Predictions generated")

if __name__ == "__main__":
    predict()