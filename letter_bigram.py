import polars as pl
from collections import Counter
import string
import json
from pathlib import Path

# Define input and output directories
input_dir = Path("data/all_names")
feature_store_path = Path("data/features")
unique_words_file = feature_store_path / "unique_words.json"

# Create feature store directory if it doesn't exist
feature_store_path.mkdir(parents=True, exist_ok=True)

# Define functions for letter & bigram frequencies
def letter_frequency(word):
    word = word.lower()
    letters = string.ascii_lowercase
    count = Counter(word)
    return {letter: count[letter] for letter in count}  # Only store present letters

def bigram_frequency(word):
    word = word.lower()
    bigrams = [word[i:i+2] for i in range(len(word)-1)]
    count = Counter(bigrams)
    return dict(count)

# Step 1: Extract Unique Words from CSV Files
unique_words = set()
for file in input_dir.glob("*.csv"):
    print(f"Extracting unique words from: {file}")
    with open(file, "r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            word = line.strip().split(",")[0]  # Assume word is the first column
            unique_words.add(word)
    break

# Save unique words for future use
with open(unique_words_file, "w") as f:
    json.dump(list(unique_words), f)
print("✅ Unique words saved at", unique_words_file)

# Step 2: Generate Features Only Once for Unique Words
feature_dict = {
    word: {
        "letters_freq": letter_frequency(word),
        "bigrams_freq": bigram_frequency(word)
    }
    for word in unique_words
}

# Save features as JSON for future use
feature_store_file = feature_store_path / "letter_bigram.json"
with open(feature_store_file, "w") as f:
    json.dump(feature_dict, f)

print("✅ Feature dictionary saved for future use at", feature_store_file)