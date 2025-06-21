import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os

# Initialize the embedding model
embedding_model = SentenceTransformer("./all-MiniLM-L6-v2")

# Path to the data file
data_file = "data/data.jsonl"
index_file = "faiss_index.bin"
doc_map_file = "documents.json"

# Read the data
documents = []
ids = []

print("Reading data...")
with open(data_file, "r") as f:
    for line in f:
        item = json.loads(line)
        documents.append(item["text"])
        ids.append(item["id"])

print("Generating embeddings...")
# Generate embeddings
embeddings = embedding_model.encode(documents, convert_to_numpy=True)

# Create a FAISS index
d = embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)
index = faiss.IndexIDMap(index) # Map vectors to original document IDs

print("Adding embeddings to FAISS index...")
# The IDs for FAISS must be integers, so we'll use the position in the list
faiss_ids = np.array(range(len(documents)))
index.add_with_ids(embeddings, faiss_ids)

print(f"Saving FAISS index to {index_file}...")
# Save the index to a file
faiss.write_index(index, index_file)

print(f"Saving document map to {doc_map_file}...")
# Save the mapping from faiss_ids (0, 1, 2...) to original docs
doc_map = {i: {"id": ids[i], "text": documents[i]} for i in range(len(documents))}
with open(doc_map_file, 'w') as f:
    json.dump(doc_map, f)

print("Data has been ingested into FAISS.") 