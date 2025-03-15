import torch
import json
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

# Paths
feature_store_path = Path("data/features")
embeddings_file = feature_store_path / "word_embeddings.json"

# Load tokenizer and model
MODEL_NAME = "DTAI-KULeuven/robbert-2023-dutch-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load word data
json_file = feature_store_path / "word_ratings_evil.json"
with open(json_file, "r") as f:
    word_data = json.load(f)

unique_words = list(word_data.keys())  # Get all unique words
embeddings_dict = {}

# Generate embeddings and store them
for word in unique_words:
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    embeddings_dict[word] = embedding.tolist()

# Save embeddings to a JSON file
with open(embeddings_file, "w") as f:
    json.dump(embeddings_dict, f)

print(f"✅ Word embeddings saved to {embeddings_file}")
