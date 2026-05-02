import pandas as pd
from newsapi import NewsApiClient
from textblob import TextBlob

# 🔑 Replace with your API key
NEWS_API_KEY = "257a94bb06b14a3a8e25797a17b79fe8"

newsapi = NewsApiClient(api_key=NEWS_API_KEY)


def get_sentiment(company_name):
    try:
        articles = newsapi.get_everything(
            q=company_name,
            language="en",
            sort_by="relevancy",
            page_size=20
        )

        sentiments = []

        for article in articles["articles"]:
            text = article["title"] + " " + (article["description"] or "")
            polarity = TextBlob(text).sentiment.polarity
            sentiments.append(polarity)

        if len(sentiments) == 0:
            return 3  # neutral fallback

        avg_sentiment = sum(sentiments) / len(sentiments)

        # Convert to risk scale (1–5)
        if avg_sentiment > 0.2:
            return 1   # positive → low risk
        elif avg_sentiment > 0:
            return 2
        elif avg_sentiment > -0.2:
            return 3
        elif avg_sentiment > -0.5:
            return 4
        else:
            return 5   # very negative → high risk

    except Exception as e:
        print(f"Error fetching sentiment for {company_name}: {e}")
        return 3


def update_sentiment():
    df = pd.read_csv("data/external/qualitative_data.csv")

    sentiment_scores = []

    for company in df["Company"]:
        name = company.split(".")[0]  # clean ticker
        score = get_sentiment(name)
        sentiment_scores.append(score)

    df["Public_Perception"] = sentiment_scores

    df.to_csv("data/external/qualitative_data.csv", index=False)

    print("Sentiment updated successfully")


if __name__ == "__main__":
    update_sentiment()