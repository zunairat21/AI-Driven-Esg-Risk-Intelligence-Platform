import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.abspath(".")
)

import pandas as pd 
from src.nlp.finbert_sentiment import(FinBERTSentimentAnalyzer)

# Initialize FinBERT analyzer
analyzer = FinBERTSentimentAnalyzer()


#Load ESG news dataset
df = pd.read_csv("data/ESG_daily_news.csv")

#Check dataset columns
print(df.columns)

#Select first 10 rows for testing
df = df.head(10)

#Store predictions 
sentiments = []
confidence_scores= []

#Loop through news headline

for text in df["headline"]:

    result = analyzer.predict_sentiment(text)
    sentiments.append(
        result["sentiment"]
        )
    
    confidence_scores.append(
        result["confidence"]
        )
    
#Add new columns
df["sentiment"]=sentiments
df["confidence"]=confidence_scores

#Display result
print(df[[
    "headline",
    "sentiment",
    "confidence"
]])

#Save processed dataset
df.to_csv("data/esg_news_with_sentment.csv", index=False)

print("Sentiment scoring completed successfully")

