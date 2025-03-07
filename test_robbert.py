import torch
from transformers import AutoModel, AutoTokenizer
import json
import numpy as np
from pathlib import Path

# Paths to saved models and embeddings
feature_store_path = Path("data/features")
embeddings_file = feature_store_path / "word_embeddings.json"
output_model_file = feature_store_path / "linear_models.json"

# Load saved embeddings
with open(embeddings_file, "r") as f:
    embeddings_dict = json.load(f)

# Load trained models
with open(output_model_file, "r") as f:
    models = json.load(f)

# Load RobBERT for new word embedding generation
MODEL_NAME = "pdelobelle/robbert-v2-dutch-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to extract embeddings from RobBERT
def get_embedding(word):
    if word in embeddings_dict:
        return embeddings_dict[word]  # Use cached embedding if available
    
    print(f"🔄 Generating embedding for '{word}'")
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    
    # Save to dictionary for future use
    embeddings_dict[word] = embedding.tolist()
    
    return embedding.tolist()

# Function to predict traits
def predict_traits(word):
    embedding = get_embedding(word)  # Get embedding
    
    # Convert embedding to numpy array
    embedding_array = np.array(embedding).reshape(1, -1)
    
    predictions = {}
    for trait, coef in models.items():
        predictions[trait] = np.dot(coef, embedding_array.T).item()
    
    return predictions

# Example usage: Predict for new words
new_words = ["Whiski", "Shell", "Amazon", "Google", "Facebook", "McDonalds", "NeoCru"]

for word in new_words:
    predictions = predict_traits(word)
    print(f"🔍 Predicted traits for {word}: {predictions}")

# Save updated embeddings with new words
with open(embeddings_file, "w") as f:
    json.dump(embeddings_dict, f)
print("✅ Updated embeddings saved for future use.")
