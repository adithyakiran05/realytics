import pandas as pd
from newsapi import NewsApiClient
from transformers import pipeline
from datetime import datetime

NEWS_API_KEY = "257a94bb06b14a3a8e25797a17b79fe8"

newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Load FinBERT
sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)


def get_weighted_sentiment(company_name):
    try:
        articles = newsapi.get_everything(
            q=company_name,
            language="en",
            sort_by="publishedAt",
            page_size=20
        )

        weighted_scores = []
        weights = []

        for article in articles["articles"]:
            text = article["title"] + " " + (article["description"] or "")

            result = sentiment_model(text[:512])[0]

            label = result["label"]
            score = result["score"]

            # Convert label to numeric
            if label == "positive":
                polarity = 1 * score
            elif label == "negative":
                polarity = -1 * score
            else:
                polarity = 0

            # Time weighting (recent news matters more)
            published = article["publishedAt"]
            date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
            days_old = (datetime.utcnow() - date).days

            time_weight = 1 / (1 + days_old)

            weighted_scores.append(polarity * time_weight)
            weights.append(time_weight)

        if not weights:
            return 3, 0.5  # neutral, low confidence

        avg = sum(weighted_scores) / sum(weights)

        # Convert to 1–5 scale
        if avg > 0.3:
            score = 1
        elif avg > 0.1:
            score = 2
        elif avg > -0.1:
            score = 3
        elif avg > -0.3:
            score = 4
        else:
            score = 5

        # Confidence = number of articles + consistency
        confidence = min(len(weights) / 20, 1.0)

        return score, confidence

    except Exception as e:
        print(f"Error: {e}")
        return 3, 0.3


def update_sentiment():
    df = pd.read_csv("data/external/qualitative_data.csv")

    perceptions = []
    confidences = []

    for company in df["Company"]:
        name = company.split(".")[0]
        score, conf = get_weighted_sentiment(name)

        perceptions.append(score)
        confidences.append(conf)

    df["Public_Perception"] = perceptions
    df["Sentiment_Confidence"] = confidences

    df.to_csv("data/external/qualitative_data.csv", index=False)

    print("Advanced sentiment updated")


if __name__ == "__main__":
    update_sentiment()