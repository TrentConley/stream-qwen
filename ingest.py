import chromadb
from sentence_transformers import SentenceTransformer
import json

# Initialize ChromaDB client and create a collection
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("documents")

# Initialize the embedding model
embedding_model = SentenceTransformer("./all-MiniLM-L6-v2")

# Path to the data file
data_file = "data/data.jsonl"

# Read the data and ingest it into ChromaDB
documents = []
metadatas = []
ids = []

with open(data_file, "r") as f:
    for line in f:
        item = json.loads(line)
        # For simplicity, we'll embed the 'text' field.
        # You can create a more sophisticated representation of your data
        # by combining the paragraph, list, and table.
        documents.append(item["text"])
        metadatas.append({"id": item["id"]})
        ids.append(item["id"])

# Generate embeddings
embeddings = embedding_model.encode(documents)

# Add to the collection
collection.add(
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("Data has been ingested into ChromaDB.")

# Example of how to query the collection
query_text = "What is RAG?"
query_embedding = embedding_model.encode([query_text])[0].tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print("Example query results:")
print(results) 