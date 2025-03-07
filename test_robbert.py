import torch
import json
import numpy as np
from pathlib import Path
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Define model structure
class TraitPredictor(nn.Module):
    def __init__(self, input_dim):
        super(TraitPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)  # Single output for regression

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load trained models for all four traits
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
traits = ["evil", "trustworthy", "smart", "feminine"]
models = {}

for trait in traits:
    model_path = f"data/features/trained_{trait}_predictor.pth"
    model = TraitPredictor(input_dim=768).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    models[trait] = model

# Load tokenizer and model for embedding generation
MODEL_NAME = "pdelobelle/robbert-v2-dutch-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embedding_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# Generate embeddings for new words without updating the stored embeddings
def get_embedding(word):
    print(f"🔄 Generating embedding for '{word}'")
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return torch.tensor(embedding, dtype=torch.float32).to(device)

# Test words for prediction
test_words = ["Tesla", "Amazon", "Google", "Facebook", "Shell", "NeoCru", "Sazelime"]

# Run predictions
for word in test_words:
    embedding = get_embedding(word).unsqueeze(0)  # Add batch dimension
    word_predictions = {}
    for trait, model in models.items():
        with torch.no_grad():  # Disable gradients for faster inference
            word_predictions[trait] = model(embedding).item()
    
    print(f"🔍 {word}: {word_predictions}")
