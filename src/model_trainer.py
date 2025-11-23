import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import FINBERT_MODEL, DATA_DIR, COMPANIES, TICKERS

class SentimentDataset(Dataset):
    def __init__(self, texts, sentiments, tokenizer, max_len):
        self.texts = texts
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        sentiment = self.sentiments[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment': torch.tensor(sentiment, dtype=torch.float)
        }

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden[-1])

class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(self.sequences[idx], dtype=torch.float),
            'target': torch.tensor(self.targets[idx], dtype=torch.float)
        }

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return self.fc(hidden[-1])

def train_sentiment_model():
    df = pd.read_csv(os.path.join(DATA_DIR, 'news_with_sentiment.csv'))
    df = df.fillna('')
    df['text'] = df['title'].astype(str) + ' ' + df['description'].astype(str) + ' ' + df['content'].astype(str)
    texts = df['text'].tolist()
    sentiments = df['sentiment'].tolist()

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    vocab_size = tokenizer.vocab_size
    max_len = 512

    X_train, X_test, y_train, y_test = train_test_split(texts, sentiments, test_size=0.2, random_state=42)

    train_dataset = SentimentDataset(X_train, y_train, tokenizer, max_len)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = LSTMModel(vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['sentiment'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['sentiment'].to(device)
                outputs = model(input_ids)
                test_preds.extend(outputs.squeeze().cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

        mse = mean_squared_error(test_targets, test_preds)
        print(f"Epoch {epoch+1}/{epochs}, MSE: {mse}")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/sentiment_model.pth')
    torch.save(tokenizer, 'models/tokenizer.pth')  # Save tokenizer for inference
    print("Sentiment model trained and saved.")

def train_stock_models():
    df = pd.read_csv(os.path.join(DATA_DIR, 'stocks.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Company', 'Date'])
    
    features = ['Open', 'High', 'Low', 'Volume']
    target = 'Close'
    seq_len = 10

    for company in TICKERS:
        company_df = df[df['Company'] == company].copy()
        if len(company_df) < seq_len + 1:
            continue

        # Normalize features per company (keep target scaling within same scaler for compatibility)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        company_df[features + [target]] = scaler.fit_transform(company_df[features + [target]])

        sequences = []
        targets = []
        for i in range(len(company_df) - seq_len):
            seq = company_df.iloc[i:i+seq_len][features].values.astype(np.float32)
            targ = company_df.iloc[i+seq_len][target].astype(np.float32)
            sequences.append(seq)
            targets.append(targ)

        if not sequences:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )

        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        input_dim = len(features)
        model = StockLSTM(input_dim, hidden_dim=64, output_dim=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        epochs = 10
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                seq = batch['sequence'].to(device)
                targ = batch['target'].to(device)
                optimizer.zero_grad()
                outputs = model(seq)
                loss = criterion(outputs.squeeze(), targ)
                loss.backward()
                optimizer.step()

            model.eval()
            test_preds = []
            test_targets = []
            with torch.no_grad():
                for batch in test_loader:
                    seq = batch['sequence'].to(device)
                    targ = batch['target'].to(device)
                    outputs = model(seq)
                    test_preds.extend(outputs.squeeze().cpu().numpy())
                    test_targets.extend(targ.cpu().numpy())

            mse = mean_squared_error(test_targets, test_preds)
            print(f"{company} Epoch {epoch+1}/{epochs}, MSE: {mse:.6f}")

        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/{company}_stock_model.pth')
        # Save the scaler for inverse transform
        import joblib
        joblib.dump(scaler, f'models/{company}_scaler.pkl')

    print("Stock models trained and saved.")

def train_models():
    #train_sentiment_model()
    train_stock_models()

if __name__ == "__main__":
    train_models()