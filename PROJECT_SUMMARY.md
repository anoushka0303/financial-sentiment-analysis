# Financial Analysis Chatbot - Complete System

## 🏗️ System Architecture

### Backend (FastAPI)
- **Location**: `/backend/`
- **Port**: 8000
- **Features**: 
  - RESTful API with authentication
  - Integration with financial analysis pipeline
  - Chat session management
  - Real-time stock data via yfinance
  - News retrieval from vector database
  - Sentiment analysis and price prediction

### Frontend (React + Vite)
- **Location**: `/frontend/`
- **Port**: 3000
- **Features**:
  - Modern chat interface
  - Dark/light mode support
  - Real-time messaging
  - Company selection
  - Responsive design

### Financial Analysis Pipeline
- **Location**: `/src/`
- **Components**:
  - yfinance integration for real-time data
  - LSTM models for price prediction
  - FinBERT for sentiment analysis
  - OpenAI GPT for comprehensive analysis
  - ChromaDB for news retrieval

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Demo Login
- **Username**: demo
- **Password**: demo

## 📊 Supported Companies
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

## 🔧 API Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `GET /auth/me` - Get current user

### Analysis
- `POST /api/analyze` - Analyze financial query
- `GET /api/companies` - List available companies
- `POST /api/chat` - Chat interface

### System
- `GET /health` - Health check
- `GET /api/analytics/dashboard` - Dashboard data

## 🎯 Key Features

1. **Real-time Stock Analysis**: Live data from yfinance
2. **News Sentiment Analysis**: FinBERT + rule-based fallback
3. **Price Prediction**: LSTM neural networks
4. **AI-Powered Insights**: OpenAI GPT integration
5. **Modern Web Interface**: React with Tailwind CSS
6. **Session Management**: Persistent chat history
7. **Authentication**: JWT-based security
8. **Responsive Design**: Mobile-friendly interface

## 🔄 Analysis Pipeline

```
User Query → Company Detection → Stock Data (yfinance) 
→ News Retrieval (ChromaDB) → Sentiment Analysis (FinBERT) 
→ Price Prediction (LSTM) → AI Analysis (GPT) → Response
```

## 🛡️ Security Features

- JWT authentication
- Password hashing (bcrypt)
- Rate limiting
- Input validation
- CORS protection
- Error handling

## 📱 User Experience

- Intuitive chat interface
- Real-time typing indicators
- Loading states and error messages
- Company selection dropdown
- Chat history persistence
- Professional dark/light themes

## 🚀 Production Ready

- Scalable architecture
- Proper error handling
- Performance optimizations
- Security best practices
- Comprehensive documentation
- Environment configuration

This system provides a complete, production-ready financial analysis chatbot with modern web technologies and robust AI capabilities.