import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR

def merge_data():
    stocks_df = pd.read_csv(os.path.join(DATA_DIR, 'stocks.csv'))
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date']).dt.date
    
    news_df = pd.read_csv(os.path.join(DATA_DIR, 'news_with_sentiment.csv'))
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt']).dt.date
    
    # Group news by company and date, average sentiment
    sentiment_avg = news_df.groupby(['company', 'publishedAt'])['sentiment'].mean().reset_index()
    sentiment_avg.rename(columns={'publishedAt': 'Date', 'company': 'Company'}, inplace=True)
    
    # Merge
    merged_df = pd.merge(stocks_df, sentiment_avg, on=['Company', 'Date'], how='inner')
    merged_df = merged_df.dropna()  # or some default
    
    merged_df.to_csv(os.path.join(DATA_DIR, 'merged_data.csv'), index=False)
    print("Data merged and saved.")

if __name__ == "__main__":
    merge_data()