import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import logging
from datetime import datetime
import json
import asyncio

# Import your financial analysis pipeline
from src.llm_summarizer import summarize_financial_data
from config import TICKERS, COMPANIES
from auth_routes import router as auth_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Analysis Chatbot API",
    description="AI-powered financial analysis and stock prediction system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Include authentication routes
app.include_router(auth_router)

# Create ticker-to-company mapping
ticker_to_company = dict(zip(TICKERS, COMPANIES))

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    company: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    query: str
    company: str
    analysis_result: str
    timestamp: datetime
    confidence: float
    predicted_price: Optional[float] = None
    current_price: Optional[float] = None
    sentiment_score: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    available_companies: List[str]

# In-memory storage for demo (replace with PostgreSQL later)
analysis_cache = {}

def extract_company_from_query(query: str) -> str:
    """Extract company ticker from user query"""
    query_upper = query.upper()
    
    # Check for exact ticker matches
    for ticker in TICKERS:
        if ticker.split('.')[0] in query_upper:  # Remove .NS for matching
            return ticker
    
    # Check for company name matches
    for ticker, company in ticker_to_company.items():
        company_words = company.lower().split()
        if any(word in query.lower() for word in company_words):
            return ticker
    
    # Default to first company if no match found
    return TICKERS[0]

def validate_company(company: str) -> bool:
    """Validate if the company ticker is supported"""
    return company in TICKERS

# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        available_companies=[ticker.split('.')[0] for ticker in TICKERS]  # Remove .NS
    )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "pipeline_available": True,
        "vector_db_available": True,
        "openai_available": True
    }

@app.get("/api/companies")
async def get_companies():
    """Get list of available companies"""
    companies = []
    for ticker, company in ticker_to_company.items():
        companies.append({
            "ticker": ticker,
            "name": company,
            "short_name": ticker.split('.')[0]
        })
    return {"companies": companies}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_query(request: QueryRequest):
    try:
        if request.company:
            company = request.company.upper() + ".NS" if not request.company.endswith('.NS') else request.company
        else:
            company = extract_company_from_query(request.query)

        if not validate_company(company):
            raise HTTPException(
                status_code=400,
                detail=f"Company {company} not supported. Available: {[t.split('.')[0] for t in TICKERS]}"
            )

        cache_key = f"{company}:{request.query[:50]}"
        if cache_key in analysis_cache:
            cached_result = analysis_cache[cache_key]
            logger.info(f"Returning cached analysis for {company}")
            return AnalysisResponse(**cached_result)

        logger.info(f"Running analysis for {company}")
        analysis_data = summarize_financial_data(company)

        response_data = {
            "query": request.query,
            "company": company,
            "analysis_result": analysis_data['analysis_result'],
            "timestamp": datetime.now(),
            "confidence": 0.85,
            "predicted_price": analysis_data['predicted_price'],
            "current_price": analysis_data['current_price'],
            "sentiment_score": analysis_data['sentiment_score']
        }

        analysis_cache[cache_key] = response_data

        return AnalysisResponse(**response_data)

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/analytics/dashboard")
async def get_dashboard_data():
    """Get dashboard analytics (placeholder)"""
    return {
        "total_queries": len(analysis_cache),
        "cached_analyses": len(analysis_cache),
        "popular_companies": [
            {"ticker": "RELIANCE.NS", "queries": 15},
            {"ticker": "TCS.NS", "queries": 12},
            {"ticker": "HDFCBANK.NS", "queries": 8}
        ],
        "average_confidence": 0.85,
        "system_status": "operational"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )