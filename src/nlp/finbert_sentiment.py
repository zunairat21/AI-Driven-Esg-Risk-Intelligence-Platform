from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
import torch.nn.functional as F


class FinBERTSentimentAnalyzer:

    def __init__(self):

        self.model_name = "ProsusAI/finbert"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        # Sentiment labels
        self.labels = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

    def predict_sentiment(self, text):

        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Run inference
        with torch.no_grad():

            outputs = self.model(**inputs)

        # Convert logits → probabilities
        probabilities = F.softmax(
            outputs.logits,
            dim=1
        )

        # Highest probability class
        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        # Confidence score
        confidence_score = probabilities[
            0,
            predicted_class
        ].item()

        return {
            "sentiment": self.labels[predicted_class],
            "confidence": round(confidence_score, 4)
        }

