import json
import random
from pathlib import Path

# Define paths
feature_store_path = Path("data/features")
unique_words_file = feature_store_path / "unique_words.json"
shuffled_words_file = feature_store_path / "shuffled_unique_words.json"

# Load unique words
with open(unique_words_file, "r") as f:
    unique_words = json.load(f)

# Shuffle the words randomly
random.seed(42)  # Set seed for reproducibility (remove this line for full randomness)
random.shuffle(unique_words)

# Save shuffled unique words
with open(shuffled_words_file, "w") as f:
    json.dump(unique_words, f)

print(f"✅ Shuffled unique words saved to {shuffled_words_file}")