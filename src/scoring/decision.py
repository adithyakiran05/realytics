import pandas as pd

def run_decision():

    df = pd.read_csv("data/processed/predictions.csv")

    # ----------------------
    # CLEAN DATA (IMPORTANT)
    # ----------------------
    df = df.fillna(0)

    if "Sentiment_Confidence" not in df.columns:
        df["Sentiment_Confidence"] = 0.5  # neutral default

    # ----------------------
    # UPSIDE SCORE
    # ----------------------
    df["Upside_Score"] = (
        0.25 * df["Revenue_Growth"] +
        0.20 * df["EBITDA_Margin"] +
        0.15 * df["ROCE"] +
        0.20 * df["Market_Growth"] +
        0.20 * df["Predicted_Growth"]
    )

    # ----------------------
    # RISK SCORE
    # ----------------------
    df["Risk_Score"] = (
        0.25 * df["Debt_Ratio"] +
        0.15 * df["Volatility"] +
        0.15 * df["Competition"] +
        0.15 * df["Industry_Risk"] +
        0.15 * df["Public_Perception"] * (1 + (1 - df["Sentiment_Confidence"]))
    )

    # ----------------------
    # DECISION SCORE
    # ----------------------
    df["Decision_Score"] = df["Upside_Score"] - df["Risk_Score"]

    # ✅ NORMALIZE (VERY IMPORTANT FOR FILTERS)
    if df["Decision_Score"].max() != df["Decision_Score"].min():
        df["Decision_Score"] = (
            df["Decision_Score"] - df["Decision_Score"].min()
        ) / (df["Decision_Score"].max() - df["Decision_Score"].min())

    # ----------------------
    # LABELING
    # ----------------------
    def label(x):
        if x > 0.6:
            return "Strong Buy"
        elif x > 0.3:
            return "Consider"
        else:
            return "Avoid"

    df["Decision"] = df["Decision_Score"].apply(label)

    # ----------------------
    # RISK EXPLANATION
    # ----------------------
    def risk_reason(row):
        reasons = []

        if row["Debt_Ratio"] > 0.5:
            reasons.append("High debt")
        if row["Competition"] >= 4:
            reasons.append("High competition")
        if row["Public_Perception"] >= 3:
            reasons.append("Weak perception")

        return ", ".join(reasons)

    df["Risk_Reason"] = df.apply(risk_reason, axis=1)

    # ----------------------
    # SAVE OUTPUT
    # ----------------------
    df.to_csv("data/processed/final_output.csv", index=False)

    print("Decision pipeline completed")


if __name__ == "__main__":
    run_decision()