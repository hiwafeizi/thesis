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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    models[trait] = model

# Load test set ratings (new unseen data)
test_data_path = Path("data/test")
word_ratings_test = {}

for trait in traits:
    test_file = test_data_path / f"word_ratings_{trait}.json"
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            word_ratings_test[trait] = json.load(f)
    else:
        print(f"⚠️ Warning: Test file for {trait} not found, skipping evaluation.")
        word_ratings_test[trait] = {}

# Load precomputed embeddings
feature_store_path = Path("data/features")
with open(feature_store_path / "word_embeddings.json", "r", encoding="utf-8") as f:
    embeddings_dict = json.load(f)

# Store model performance and predictions on the test set
performance_test_file = feature_store_path / "model_performance_test.json"
predictions_file = feature_store_path / "word_predictions_test.json"

performance_results_test = {}
predictions_dict = {}

# Process each trait separately to manage memory
for trait in traits:
    print(f"🔄 Evaluating on unseen test data for trait: {trait}")

    if not word_ratings_test[trait]:
        print(f"⚠️ Skipping {trait} due to missing test data.")
        continue  # Skip if no test data is available

    y_true = []
    y_pred = []
    batch_size = 200  # Process in chunks to reduce memory load
    words = list(word_ratings_test[trait].keys())

    predictions_dict[trait] = {}

    for i in range(0, len(words), batch_size):
        batch_words = words[i:i + batch_size]
        for word in batch_words:
            if word in embeddings_dict:
                embedding = torch.tensor(embeddings_dict[word], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = models[trait](embedding).item()
                
                # Store predictions per word
                predictions_dict[trait][word] = prediction

                y_true.extend(word_ratings_test[trait][word])  # Extend with actual ratings
                y_pred.extend([prediction] * len(word_ratings_test[trait][word]))  # Same prediction for each variation

    # Compute performance metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    performance_results_test[trait] = {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr
    }

    print(f"📊 Test Performance for {trait}:")
    print(f"✅ Mean Squared Error (MSE): {mse:.4f}")
    print(f"✅ Mean Absolute Error (MAE): {mae:.4f}")
    print(f"✅ R² Score: {r2:.4f}")
    print(f"✅ Pearson Correlation: {pearson_corr:.4f}")
    print(f"✅ Spearman Correlation: {spearman_corr:.4f}")

# Save performance results
with open(performance_test_file, "w", encoding="utf-8") as f:
    json.dump(performance_results_test, f, indent=4)

# Save individual word predictions
with open(predictions_file, "w", encoding="utf-8") as f:
    json.dump(predictions_dict, f, indent=4)

print(f"✅ Test set evaluation complete. Results saved in {performance_test_file}")
print(f"✅ Individual word predictions saved in {predictions_file}")
