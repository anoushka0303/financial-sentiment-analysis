from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from config import FINBERT_MODEL, DATA_DIR
import os

def analyze_sentiment():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    
    df = pd.read_csv(os.path.join(DATA_DIR, 'news.csv'))
    sentiments = []
    
    for idx, row in df.iterrows():
        text = f"{row['title']} {row['description']} {row['content']}"
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Assuming labels: 0 negative, 1 neutral, 2 positive
        sentiment = probs[0][2].item() - probs[0][0].item()  # positive - negative
        sentiments.append(sentiment)
    
    df['sentiment'] = sentiments
    df.to_csv(os.path.join(DATA_DIR, 'news_with_sentiment.csv'), index=False)
    print("Sentiment analysis completed.")

if __name__ == "__main__":
    analyze_sentiment()