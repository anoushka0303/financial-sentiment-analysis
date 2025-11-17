import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model Settings
FINBERT_MODEL = 'ProsusAI/finbert'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'gpt-3.5-turbo'  

# Data Settings
TICKERS = ["RELIANCE.NS","HDFCBANK.NS","BHARTIARTL.NS","TCS.NS","ICICIBANK.NS","SBIN.NS","HINDUNILVR.NS","INFY.NS","BAJFINANCE.NS","LICI.NS"]
COMPANIES = COMPANIES = [
    "Reliance Industries",
    "HDFC Bank",
    "Bharti Airtel",
    "Tata Consultancy Services",
    "ICICI Bank",
    "State Bank of India",
    "Hindustan Unilever",
    "Infosys",
    "Bajaj Finance",
    "Life Insurance Corporation of India"
]
START_DATE = '2014-01-01'
END_DATE = '2026-01-01'
PERIOD = '1d'

# Paths
DATA_DIR = 'data/'
EMBEDDINGS_DIR = 'embeddings/'
VECTOR_DB_PATH = 'embeddings/vector_db'

# Vector DB Settings
COLLECTION_NAME = 'financial_news'