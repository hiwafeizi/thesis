import torch
import json
import numpy as np
from pathlib import Path
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

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

# Load stored ratings for evaluation
feature_store_path = Path("data/features")
word_ratings = {}
for trait in traits:
    with open(feature_store_path / f"word_ratings_{trait}.json", "r") as f:
        word_ratings[trait] = json.load(f)

# Load precomputed embeddings
with open(feature_store_path / "word_embeddings.json", "r") as f:
    embeddings_dict = json.load(f)

# Store model performance
performance_file = feature_store_path / "model_performance.json"
performance_results = {}

# Process each trait separately to manage memory
for trait in traits:
    print(f"🔄 Processing trait: {trait}")
    
    # Load existing performance data if available
    if performance_file.exists():
        with open(performance_file, "r") as f:
            performance_results = json.load(f)
    
    y_true = []
    y_pred = []
    batch_size = 200  # Process in chunks to reduce memory load
    words = list(word_ratings[trait].keys())
    
    for i in range(0, len(words), batch_size):
        batch_words = words[i:i + batch_size]
        for word in batch_words:
            if word in embeddings_dict:
                embedding = torch.tensor(embeddings_dict[word], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = models[trait](embedding).item()
                y_true.extend(word_ratings[trait][word])  # Extend with actual ratings
                y_pred.extend([prediction] * len(word_ratings[trait][word]))  # Same prediction for each variation
    
    # Compute performance metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    performance_results[trait] = {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr
    }
    
    print(f"📊 Performance for {trait}:")
    print(f"✅ Mean Squared Error (MSE): {mse:.4f}")
    print(f"✅ Mean Absolute Error (MAE): {mae:.4f}")
    print(f"✅ R² Score: {r2:.4f}")
    print(f"✅ Pearson Correlation: {pearson_corr:.4f}")
    print(f"✅ Spearman Correlation: {spearman_corr:.4f}")
    
    # Save results after processing each trait
    with open(performance_file, "w") as f:
        json.dump(performance_results, f, indent=4)
    print(f"✅ Updated model performance results saved to {performance_file}")
