import sys
import os 
#Add project root to path 
sys.path.append(os.path.abspath("."))

from src.nlp.finbert_sentiment import FinBERTSentimentAnalyzer

#Initialize Analyzer

analyzer = FinBERTSentimentAnalyzer()

#Example ESG_related text

sample_text = """
Tesla reported strong quarterly revenue growth
with increasing investor confidence.
"""

#Predict sentiment
result = analyzer.predict_sentiment(sample_text)

#Print_result
print("Sentiment Analysis Result:")
print(result)
