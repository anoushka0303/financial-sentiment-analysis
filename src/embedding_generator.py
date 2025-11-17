from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import numpy as np
from config import EMBEDDING_MODEL, VECTOR_DB_PATH, COLLECTION_NAME, DATA_DIR
import os

def generate_embeddings():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    df = pd.read_csv(os.path.join(DATA_DIR, 'news_with_sentiment.csv'))
    
    documents = []
    embeddings = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        text = f"{row['title']} {row['description']} {row['content']}"
        embedding = model.encode(text).tolist()
        
        documents.append(text)
        embeddings.append(embedding)
        metadatas.append({
            'company': row['company'],
            'date': row['publishedAt'],
            'sentiment': str(row['sentiment'])
        })
        ids.append(str(idx))
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print("Embeddings generated and added to vector DB.")

if __name__ == "__main__":
    generate_embeddings()