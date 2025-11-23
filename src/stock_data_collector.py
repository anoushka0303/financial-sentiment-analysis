import yfinance as yf
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import TICKERS, COMPANIES, START_DATE, END_DATE, PERIOD, DATA_DIR

def collect_stock_data():
    all_data = []
    for company in TICKERS:
        print(f"Collecting data for {company}")
        ticker = yf.Ticker(company)
        data = ticker.history(start=START_DATE, end=END_DATE, interval=PERIOD)
        data['Company'] = company
        data.reset_index(inplace=True)
        all_data.append(data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    combined_data.to_csv(os.path.join(DATA_DIR, 'stocks.csv'), index=False)
    print("Stock data collected and saved.")

if __name__ == "__main__":
    collect_stock_data()