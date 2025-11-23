import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
from config import DATA_DIR, START_DATE, END_DATE

# Map filenames to tickers
file_to_ticker = {
    "reliance_industries_ri_news.csv": "RELIANCE.NS",
    "hdfc_bank_hdf01_news.csv": "HDFCBANK.NS",
    "bharti_airtel_btv_news.csv": "BHARTIARTL.NS",
    "tata_consultancy_services_tcs_news.csv": "TCS.NS",
    "icici_bank_ici02_news.csv": "ICICIBANK.NS",
    "state_bank_of_india_sbi_news.csv": "SBIN.NS",
    "hindustan_unilever_hl_news.csv": "HINDUNILVR.NS",
    "infosys_it_news.csv": "INFY.NS",
    "bajaj_finance_baf_news.csv": "BAJFINANCE.NS",
    "lici_news.csv": "LICI.NS",  # add if needed
}

def collect_from_kaggle_folder():
    all_rows = []

    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv"):
            continue

        if file.lower() in ["stocks.csv", "news.csv"]:
            print(f"Skipping {file}")
            continue

        path = os.path.join(DATA_DIR, file)
        print("Reading:", file)

        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]

        date_col = "date"
        title_col = "title"
        desc_col = "article description"
        text_col = "article text"
        content_col = "article content"

        df[date_col] = pd.to_datetime(df.get(date_col), errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df[df[date_col].between(start, end)]

        # Get ticker from file
        ticker = file_to_ticker.get(file)
        if not ticker:
            print(f"Skipping unknown file: {file}")
            continue

        for _, row in df.iterrows():
            all_rows.append({
                "publishedAt": row[date_col].isoformat(),
                "title": str(row.get(title_col, "")),
                "description": str(row.get(desc_col, "")),
                "content": str(row.get(content_col) or row.get(text_col) or ""),
                "source_file": file,
                "company": ticker
            })

    out_path = os.path.join(DATA_DIR, "news.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    collect_from_kaggle_folder()