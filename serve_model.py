import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    print("server is running")
    return {"status": "Server is running"}

# FAISS and SentenceTransformer setup
print("Loading FAISS index and document map...")
index = faiss.read_index("faiss_index.bin")
with open("documents.json", 'r') as f:
    doc_map = json.load(f)
embedding_model = SentenceTransformer("./all-MiniLM-L6-v2")
print("Loading complete.")

# Model and Tokenizer Configuration
MODEL_NAME = "./qwen3-0.6b-model"

# Determine the device to run the model on
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

class StreamRequest(BaseModel):
    """Request model for the streaming endpoint."""
    prompt: str

@app.post("/stream")
async def stream(request: StreamRequest):
    print(f"Received request: {request.prompt}")
    """
    Handles streaming text generation for a given prompt.
    """
    # Embed the prompt and query FAISS
    query_embedding = embedding_model.encode([request.prompt], convert_to_numpy=True)
    D, I = index.search(query_embedding, 10) # D: distances, I: indices
    
    # Get the documents from the results
    retrieved_indices = I[0]
    retrieved_documents = [doc_map[str(i)]['text'] for i in retrieved_indices]
    
    # Create the context string
    context = "\n".join(retrieved_documents)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Prepare the input for the model
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer the user's question:\n\n{context}"},
        {"role": "user", "content": request.prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generation arguments
    generation_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=32768,
    )

    # Start the generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def generate():
        """
        Generator function to yield decoded tokens from the streamer.
        """
        for new_text in streamer:
            yield new_text

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    # To run this server, use the command:
    # uvicorn inference_small_qwen:app --host 0.0.0.0 --port 8008
    uvicorn.run(app, host="0.0.0.0", port=8008)
