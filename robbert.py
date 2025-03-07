import torch
from transformers import AutoModel, AutoTokenizer
import polars as pl
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define paths
feature_store_path = Path("data/features")
word_features_file = feature_store_path / "word_features.json"
embeddings_file = feature_store_path / "word_embeddings.json"
output_model_file = feature_store_path / "linear_models.json"

# Load words from precomputed feature file
with open(word_features_file, "r") as f:
    word_features = json.load(f)
words = list(word_features.keys())  # Extract words from dictionary

# Load RobBERT Model and Tokenizer
MODEL_NAME = "pdelobelle/robbert-v2-dutch-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to extract embeddings from RobBERT
def get_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return embedding.tolist()

# Process embeddings in batches
embeddings_dict = {}
batch_size = 100
for i in range(0, len(words), batch_size):
    batch_words = words[i:i+batch_size]
    batch_embeddings = {word: get_embedding(word) for word in batch_words}
    embeddings_dict.update(batch_embeddings)

# Save embeddings for future use
with open(embeddings_file, "w") as f:
    json.dump(embeddings_dict, f)
print(f"✅ Word embeddings saved to {embeddings_file}")

# Load human rating dataset
rating_files = {
    "trustworthy": "data/company_names/company_names_trustworthy.csv",
    "evil": "data/company_names/company_names_evil.csv",
    "smart": "data/company_names/company_names_smart.csv",
    "feminine": "data/company_names/company_names_feminine.csv"
}

# Train linear models for each trait incrementally per batch
models = {}
batch_size = 50000  # Read files in 50,000-row chunks
for trait, file in rating_files.items():
    print(f"Processing trait: {trait}")
    model = LinearRegression()
    reader = pl.read_csv(file).iter_slices(batch_size)
    
    for df in reader:
        words_list = df["word"].to_list()
        X_batch = np.array([embeddings_dict[word] for word in words_list if word in embeddings_dict])
        y_batch = df["rating_value"].to_numpy()
        
        if X_batch.shape[0] > 0:
            model.fit(X_batch, y_batch)
            predictions = model.predict(X_batch)
            r2 = r2_score(y_batch, predictions)
            print(f"Batch R² score for {trait}: {r2}")

    models[trait] = model.coef_.tolist()

# Save trained models
with open(output_model_file, "w") as f:
    json.dump(models, f)
print(f"✅ Linear models saved to {output_model_file}")

# Function to predict ratings using trained models
def predict_traits(word):
    if word not in embeddings_dict:
        return {trait: None for trait in models.keys()}
    embedding = np.array(embeddings_dict[word]).reshape(1, -1)
    predictions = {trait: np.dot(models[trait], embedding.T).item() for trait in models.keys()}
    return predictions

# Example usage
example_word = "Tesla"
predictions = predict_traits(example_word)
print(f"Predicted traits for {example_word}: {predictions}")