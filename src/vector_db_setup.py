import chromadb
from config import VECTOR_DB_PATH, COLLECTION_NAME

def setup_vector_db():
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print("Vector DB setup completed.")
    return client, collection

if __name__ == "__main__":
    setup_vector_db()