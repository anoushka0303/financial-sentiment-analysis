import torch
import pandas as pd
from transformers import AutoTokenizer
from config import FINBERT_MODEL, DATA_DIR
import os
from src.model_trainer import LSTMModel, StockLSTM

try:
    from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
    torch.serialization.add_safe_globals([BertTokenizerFast])
except:
    pass  

def load_sentiment_model():
    try:
        tokenizer = torch.load('models/tokenizer.pth', weights_only=False)
        vocab_size = tokenizer.vocab_size
        model = LSTMModel(vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1)
        model.load_state_dict(torch.load('models/sentiment_model.pth', weights_only=False))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        # Return a simple fallback
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return None, None, device

def load_stock_model(company):
    try:
        input_dim = 5  # Open, High, Low, Volume, sentiment
        model = StockLSTM(input_dim, hidden_dim=64, output_dim=1)
        model.load_state_dict(torch.load(f'models/{company}_stock_model.pth', weights_only=False))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        print(f"Error loading stock model for {company}: {e}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return None, device

def predict_sentiment(text, model, tokenizer, device):
    if model is None or tokenizer is None:
        return 0.0  # Return neutral sentiment if model unavailable
    
    try:
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        with torch.no_grad():
            output = model(input_ids)
        return output.item()
    except Exception as e:
        print(f"Error in sentiment prediction: {e}")
        return 0.0

def predict_stock_price(company, user_query, recent_features):
    # recent_features: list of 10 dicts or lists with 'Open', 'High', 'Low', 'Volume', 'sentiment'
    # But sentiment will be replaced

    sentiment_model, tokenizer, device_sent = load_sentiment_model()
    sentiment_score = predict_sentiment(user_query, sentiment_model, tokenizer, device_sent)

    # Prepare sequence: last 10 days features, replace last sentiment with predicted
    seq = []
    for i, feat in enumerate(recent_features):
        if isinstance(feat, dict):
            row = [feat['Open'], feat['High'], feat['Low'], feat['Volume'], feat['sentiment']]
        else:
            row = feat
        if i == 9:  # last one
            row[-1] = sentiment_score
        seq.append(row)

    stock_model, device_stock = load_stock_model(company)
    if stock_model is None:
        # Return simple average prediction if model unavailable
        prices = [row[3] for row in seq]  # Use Close prices
        return sum(prices) / len(prices)
    
    try:
        seq_tensor = torch.tensor(seq, dtype=torch.float).unsqueeze(0).to(device_stock)
        with torch.no_grad():
            pred = stock_model(seq_tensor)
        return pred.item()
    except Exception as e:
        print(f"Error in stock prediction: {e}")
        # Return simple average prediction as fallback
        prices = [row[3] for row in seq]  # Use Close prices
        return sum(prices) / len(prices)

# Example usage:
# recent_features = [...]  # list of 10 feature lists
# prediction = predict_stock_price('AAPL', 'What is the sentiment for Apple stock?', recent_features)