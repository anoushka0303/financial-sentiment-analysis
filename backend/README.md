# Financial Analysis Chatbot API

AI-powered financial analysis and stock prediction system with chatbot interface.

## Features

- Real-time stock data using yfinance
- News sentiment analysis using FinBERT
- Stock price prediction with LSTM models
- Vector database for news retrieval
- OpenAI GPT integration for analysis
- User authentication and session management
- Chat history and analytics

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

3. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `GET /auth/me` - Get current user
- `POST /auth/logout` - User logout

### Analysis
- `GET /` - Health check
- `GET /health` - System health status
- `GET /api/companies` - List available companies
- `POST /api/analyze` - Analyze financial query
- `POST /api/chat` - Chat interface

### Chat Management
- `GET /api/chat/history/{session_id}` - Get chat history
- `DELETE /api/chat/history/{session_id}` - Clear chat history

### Analytics
- `GET /api/analytics/dashboard` - Dashboard data

## Configuration

Edit `config.py` for system settings:
- Server configuration
- Rate limiting
- Cache settings
- Security settings

## Environment Variables

See `.env` file for all available configurations.

## Rate Limits

- 60 requests per minute per IP
- 1000 requests per hour per IP

## Supported Companies

The system currently supports these Indian stocks:
- RELIANCE.NS (Reliance Industries)
- HDFCBANK.NS (HDFC Bank)
- TCS.NS (Tata Consultancy Services)
- ICICIBANK.NS (ICICI Bank)
- SBIN.NS (State Bank of India)
- HINDUNILVR.NS (Hindustan Unilever)
- INFY.NS (Infosys)
- BAJFINANCE.NS (Bajaj Finance)
- BHARTIARTL.NS (Bharti Airtel)
- LICI.NS (Life Insurance Corporation)

## Architecture

```
Financial Analysis Pipeline:
1. User Query → Company Extraction
2. yfinance → Latest Stock Data
3. Vector DB → Relevant News Articles
4. FinBERT → Sentiment Analysis
5. LSTM Model → Price Prediction
6. OpenAI GPT → Comprehensive Analysis
7. Response → User Interface
```

## Security

- JWT-based authentication
- Password hashing with bcrypt
- Rate limiting
- Input validation
- CORS protection

## Deployment

For production deployment:
1. Use a production WSGI server like Gunicorn
2. Set up PostgreSQL database
3. Configure reverse proxy (nginx)
4. Set up SSL certificates
5. Configure proper CORS origins