import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import VECTOR_DB_PATH, COLLECTION_NAME, DATA_DIR, EMBEDDING_MODEL, TICKERS, COMPANIES

ticker_to_company = dict(zip(TICKERS, COMPANIES))

def setup_vector_db():
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted existing collection.")
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    news_file = os.path.join(DATA_DIR, 'news.csv')
    if not os.path.exists(news_file):
        print(f"News file not found: {news_file}")
        return client, collection

    df = pd.read_csv(news_file)
    print(f"Loaded {len(df)} news articles.")

    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        text = f"{row['title']} {row['description']} {row['content']}"
        company_name = ticker_to_company.get(row['company'], row['company'])
        metadata = {
            'company': company_name,
            'ticker': row['company'],
            'publishedAt': row['publishedAt'],
            'source_file': row['source_file']
        }
        documents.append(text)
        metadatas.append(metadata)
        ids.append(f"news_{idx}")

    print("Generating embeddings...")
    embeddings = embed_model.encode(documents).tolist()

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            embeddings=embeddings[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

    print(f"Vector DB populated with {len(documents)} documents.")
    return client, collection

if __name__ == "__main__":
    setup_vector_db()