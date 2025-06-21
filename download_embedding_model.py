from sentence_transformers import SentenceTransformer
import os

model_name = 'all-MiniLM-L6-v2'
save_path = f'./{model_name}'

if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"Downloading model '{model_name}' to '{save_path}'...")
model = SentenceTransformer(model_name)
model.save(save_path)
print("Model downloaded successfully.") 