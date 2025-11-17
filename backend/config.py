import os
from dotenv import load_dotenv

# Load environment variables from root .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 1))

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Rate Limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 60))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", 1000))

# Cache Settings
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))  # 1 hour
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", 1000))

# Session Settings
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", 24))
MAX_SESSION_MESSAGES = int(os.getenv("MAX_SESSION_MESSAGES", 100))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/api.log")

# Financial Analysis Pipeline Settings
ANALYSIS_TIMEOUT_SECONDS = int(os.getenv("ANALYSIS_TIMEOUT_SECONDS", 120))
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", 5))

# Response Limits
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 500))
MAX_ANALYSIS_LENGTH = int(os.getenv("MAX_ANALYSIS_LENGTH", 5000))

# API Keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model Settings
FINBERT_MODEL = 'ProsusAI/finbert'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'gpt-3.5-turbo'

# Data Settings
START_DATE = '2014-01-01'
END_DATE = '2026-01-01'
PERIOD = '1d'

# Paths
DATA_DIR = 'data/'
EMBEDDINGS_DIR = 'embeddings/'
VECTOR_DB_PATH = 'embeddings/vector_db'

# Vector DB Settings
COLLECTION_NAME = 'financial_news'

# Company List (from main config)
TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "TCS.NS",
    "ICICIBANK.NS", "SBIN.NS", "HINDUNILVR.NS", "INFY.NS",
    "BAJFINANCE.NS", "LICI.NS"
]

COMPANIES = [
    "Reliance Industries", "HDFC Bank", "Bharti Airtel",
    "Tata Consultancy Services", "ICICI Bank", "State Bank of India",
    "Hindustan Unilever", "Infosys", "Bajaj Finance",
    "Life Insurance Corporation of India"
]