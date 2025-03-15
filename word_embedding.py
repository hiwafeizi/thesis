import torch
import json
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

# Define paths
feature_store_path = Path("data/features")
embeddings_file = feature_store_path / "word_embeddings.json"
unique_words_file = feature_store_path / "unique_words.json"  # File containing your unique words list

# Load unique words from the JSON file
with open(unique_words_file, "r", encoding="utf-8") as f:
    unique_words = json.load(f)  # Expects a JSON list, e.g., ["Kaoutar", "Midewi", "schalbier", ...]

# Load tokenizer and model for RobBERT
MODEL_NAME = "DTAI-KULeuven/robbert-2023-dutch-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embeddings_dict = {}

# Generate embeddings for each word and store them
for word in unique_words:
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    # Move input tensors to the device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the embedding from the [CLS] token (first token)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    embeddings_dict[word] = embedding.tolist()

# Save the embeddings to a JSON file
with open(embeddings_file, "w", encoding="utf-8") as f:
    json.dump(embeddings_dict, f)

print(f"✅ Word embeddings saved to {embeddings_file}")
