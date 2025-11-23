import openai
import chromadb
import torch
import pandas as pd
import yfinance as yf
from sentence_transformers import SentenceTransformer
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import OPENAI_API_KEY, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, TICKERS, COMPANIES
VECTOR_DB_PATH = '../embeddings/vector_db'
from datetime import datetime, timedelta
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

ticker_to_company = dict(zip(TICKERS, COMPANIES))

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def get_latest_stock_data(company, days=30):
    """Get latest stock data using yfinance"""
    try:
        ticker = yf.Ticker(company)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = ticker.history(start=start_date, end=end_date, interval='1d')
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        
        # Add sentiment column based on recent news analysis
        try:
            news_texts, _ = get_recent_news(company, n_results=5)
            sentiment_results = analyze_sentiment_with_finbert(news_texts)
            sentiment_score = sentiment_results['average_score']
        except Exception as e:
            print(f"Error getting sentiment for {company}: {e}")
            sentiment_score = 0.0
        data['sentiment'] = sentiment_score
        
        return data.tail(10)  # Return last 10 days for prediction
    except Exception as e:
        print(f"Error fetching data for {company}: {e}")
        return pd.DataFrame()

def get_recent_news(company, n_results=5):
    """Retrieve recent news articles from vector database"""
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        
        company_name = ticker_to_company.get(company, company)
        query_text = f"Financial news about {company_name}"
        query_embedding = embed_model.encode(query_text).tolist()
        
        # Query with company filter
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=n_results, 
            where={'company': company_name} if company_name != company else None
        )
        
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        return documents, metadatas
    except Exception as e:
        print(f"Error retrieving news for {company}: {e}")
        return [], []

def analyze_sentiment_with_finbert(news_texts):
    """Analyze sentiment using FinBERT model"""
    if not news_texts:
        return {
            'average_score': 0.0,
            'label': 'Neutral',
            'individual_scores': [0.0]
        }
    
    try:
        # Use FinBERT for sentiment analysis
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        sentiments = []
        for text in news_texts:
            # Truncate text for FinBERT (512 tokens max)
            inputs = tokenizer(text[:500], return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get probabilities for positive, negative, neutral
            probs = F.softmax(outputs.logits, dim=-1)[0]
            # FinBERT labels: 0=positive, 1=negative, 2=neutral
            positive_prob = probs[0].item()
            negative_prob = probs[1].item()
            neutral_prob = probs[2].item()
            
            # Convert to sentiment score (-1 to 1)
            sentiment_score = positive_prob - negative_prob
            sentiments.append(sentiment_score)
        
        avg_sentiment = np.mean(sentiments)
        
        if avg_sentiment > 0.1:
            sentiment_label = "Positive"
        elif avg_sentiment < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
            
        return {
            'average_score': avg_sentiment,
            'label': sentiment_label,
            'individual_scores': sentiments
        }
    except Exception as e:
        print(f"Error in FinBERT sentiment analysis: {e}")
        # Fallback to rule-based analysis
        return analyze_sentiment_rule_based(news_texts)

def analyze_sentiment_rule_based(news_texts):
    """Analyze sentiment using rule-based approach as fallback"""
    if not news_texts:
        return {
            'average_score': 0.0,
            'label': 'Neutral',
            'individual_scores': [0.0]
        }
    
    try:
        positive_words = [
            'profit', 'growth', 'increase', 'gain', 'rise', 'bullish', 'positive', 
            'strong', 'beat', 'exceed', 'upgrade', 'buy', 'recommend', 'opportunity',
            'improvement', 'expansion', 'success', 'record', 'milestone', 'breakthrough',
            'earnings', 'revenue', 'dividend', 'bonus', 'investment', 'acquisition'
        ]
        
        negative_words = [
            'loss', 'decline', 'decrease', 'fall', 'drop', 'bearish', 'negative', 
            'weak', 'miss', 'underperform', 'downgrade', 'sell', 'risk', 'concern',
            'crisis', 'recession', 'volatility', 'uncertainty', 'challenge', 'problem',
            'debt', 'bankruptcy', 'lawsuit', 'fraud', 'scandal', 'investigation'
        ]
        
        sentiments = []
        for text in news_texts:
            # Convert to lowercase for analysis
            text_lower = text.lower()
            
            # Count positive and negative words
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            # Calculate sentiment score
            total_words = len(text.split())
            sentiment_score = (positive_count - negative_count) / max(total_words, 1)
            
            # Normalize to [-1, 1] range
            sentiment_score = max(-1.0, min(1.0, sentiment_score * 10))
            sentiments.append(sentiment_score)
        
        avg_sentiment = np.mean(sentiments)
        
        if avg_sentiment > 0.1:
            sentiment_label = "Positive"
        elif avg_sentiment < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
            
        return {
            'average_score': avg_sentiment,
            'label': sentiment_label,
            'individual_scores': sentiments
        }
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {
            'average_score': 0.0,
            'label': 'Neutral',
            'individual_scores': [0.0]
        }

def load_stock_model(company):
    """Load stock prediction model for the given company"""
    try:
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler
        import joblib

        # Load the scaler
        scaler_path = f'../models/{company}_scaler.pkl'
        scaler = joblib.load(scaler_path)

        # Define model architecture (must match training)
        class StockLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                output, (hidden, cell) = self.lstm(x)
                return self.fc(hidden[-1])

        # Initialize model with correct input dimension (4 features)
        input_dim = 4  # Open, High, Low, Volume
        model = StockLSTM(input_dim, hidden_dim=64, output_dim=1)

        # Load weights
        model.load_state_dict(torch.load(f'../models/{company}_stock_model.pth', weights_only=False))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        return model, scaler, device
    except Exception as e:
        print(f"Error loading stock model for {company}: {e}")
        return None, None, None, None

def predict_stock_price(company, recent_features):
    """Predict stock price using the trained model"""
    try:
        model, scaler, device = load_stock_model(company)
        if model is None:
            return None

        # Prepare features (only 4: Open, High, Low, Volume)
        features_only = []
        for feat in recent_features:
            if isinstance(feat, dict):
                row = [feat['Open'], feat['High'], feat['Low'], feat['Volume']]
            else:
                row = feat[:4]  # Take only first 4 features
            features_only.append(row)

        # Normalize using the loaded scaler (scaler expects 5 features: 4 + target)
        features_array = np.array(features_only)  # (n, 4)
        # Add dummy target column for scaler compatibility
        dummy_features = np.column_stack([features_array, np.zeros(len(features_array))])  # (n, 5)
        features_normalized = scaler.transform(dummy_features)  # (n, 5)
        # Take only the 4 feature columns for LSTM input
        features_for_lstm = features_normalized[:, :4]  # (n, 4)

        # Convert to tensor
        seq_tensor = torch.tensor(features_for_lstm, dtype=torch.float).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(seq_tensor)

        # Inverse transform the prediction
        # We need to create a dummy array with the same shape as training data
        dummy_array = np.zeros((1, 5))  # 5 columns (4 features + target)
        dummy_array[0, -1] = prediction.item()  # Put prediction in target column
        denormalized = scaler.inverse_transform(dummy_array)

        return denormalized[0, -1]  # Return the predicted price
    except Exception as e:
        print(f"Error predicting stock price: {e}")
        return None

def prepare_prediction_features(stock_data):
    """Prepare features for stock prediction model"""
    features = []
    for _, row in stock_data.iterrows():
        feature_row = [
            float(row['Open']),
            float(row['High']),
            float(row['Low']),
            float(row['Volume'])
        ]
        features.append(feature_row)
    return features

def summarize_financial_data(company, query_date=None):
    """
    Enhanced financial data summarization with yfinance, prediction, and sentiment analysis
    Returns a dict with analysis text and structured data
    """
    try:
        company_name = ticker_to_company.get(company, company)

        # 1. Get latest stock data using yfinance
        print(f"Fetching latest stock data for {company}...")
        stock_data = get_latest_stock_data(company, days=30)
        if stock_data.empty:
            return {
                'analysis_result': f"No stock data available for {company}",
                'current_price': None,
                'predicted_price': None,
                'sentiment_score': 0.0
            }

        latest_stock = stock_data.iloc[-1]
        current_price = float(latest_stock['Close'])

        # 2. Get recent news articles from vector DB
        print(f"Retrieving recent news for {company}...")
        news_texts, news_metadatas = get_recent_news(company, n_results=5)

        # 3. Analyze sentiment of news articles
        print("Analyzing sentiment of news articles...")
        sentiment_results = analyze_sentiment_with_finbert(news_texts)

        # 4. Prepare features and predict stock price
        print("Predicting stock price...")
        recent_features = prepare_prediction_features(stock_data.tail(10))

        # Try using the stock prediction model
        try:
            predicted_price = predict_stock_price(company, recent_features)
            if predicted_price is None:
                # Simple fallback: trend-based prediction
                recent_prices = [row[0] for row in recent_features]  # Close prices (using Open as proxy)
                avg_price = sum(recent_prices) / len(recent_prices)
                predicted_price = avg_price + (sentiment_results['average_score'] * avg_price * 0.02)
        except Exception as e:
            print(f"Error in stock prediction: {e}")
            # Simple fallback: trend-based prediction
            recent_prices = [row[0] for row in recent_features]  # Close prices (using Open as proxy)
            avg_price = sum(recent_prices) / len(recent_prices)
            predicted_price = avg_price + (sentiment_results['average_score'] * avg_price * 0.02)

        # 5. Generate comprehensive prompt for LLM
        news_summary = ""
        if news_texts:
            news_summary = "\n".join([f"• {text[:200]}..." for text in news_texts[:3]])
        else:
            news_summary = "No recent news articles found."

        sentiment_summary = f"""
Sentiment Analysis:
- Average Sentiment Score: {sentiment_results['average_score']:.3f}
- Sentiment Label: {sentiment_results['label']}
- Analysis Method: FinBERT + Rule-based fallback
- Individual Scores: {[f'{score:.3f}' for score in sentiment_results['individual_scores']]}
        """.strip()

        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price * 100)

        prompt = f"""
        As a professional financial analyst, provide a comprehensive analysis of {company_name} based on the following data:

        CURRENT STOCK DATA (Latest):
        - Current Price: ${current_price:.2f}
        - Date: {latest_stock['Date']}
        - Open: ${float(latest_stock['Open']):.2f}
        - High: ${float(latest_stock['High']):.2f}
        - Low: ${float(latest_stock['Low']):.2f}
        - Volume: {int(latest_stock['Volume']):,}

        PREDICTED PRICE:
        - Next Day Predicted Closing Price: ${predicted_price:.2f}
        - Price Change Prediction: ${price_change:+.2f} ({price_change_pct:+.2f}%)
        - Prediction Confidence: {'High' if abs(sentiment_results['average_score']) > 0.3 else 'Medium' if abs(sentiment_results['average_score']) > 0.1 else 'Low'}

        RECENT NEWS ARTICLES:
        {news_summary}

        {sentiment_summary}

        Please provide:
        1. Executive summary of current situation
        2. Key factors driving the stock movement
        3. News impact analysis and interpretation
        4. Sentiment analysis interpretation and confidence
        5. Price prediction rationale and risk factors
        6. Investment recommendation (BUY/HOLD/SELL) with detailed reasoning
        7. Risk assessment and market outlook

        Focus on data-driven insights and practical investment guidance.
        """

        # 6. Get LLM response using new OpenAI API
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional financial analyst with expertise in stock market analysis, sentiment analysis, and investment recommendations. Provide clear, actionable insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.7
        )

        analysis_result = response.choices[0].message.content

        return {
            'analysis_result': analysis_result,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'sentiment_score': sentiment_results['average_score']
        }

    except Exception as e:
        print(f"Error in financial analysis for {company}: {e}")
        return {
            'analysis_result': f"Error analyzing {company}: {str(e)}",
            'current_price': None,
            'predicted_price': None,
            'sentiment_score': 0.0
        }

if __name__ == "__main__":
    # Test with different companies
    test_companies = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
    
    for company in test_companies:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {company}")
        print(f"{'='*50}")

        try:
            result = summarize_financial_data(company)
            if isinstance(result, dict):
                print(f"Current Price: ${result['current_price']:.2f}")
                print(f"Predicted Price: ${result['predicted_price']:.2f}")
                print(f"Sentiment Score: {result['sentiment_score']:.3f}")
                print(f"\nAnalysis:\n{result['analysis_result']}")
            else:
                print(result)
        except Exception as e:
            print(f"Error: {e}")