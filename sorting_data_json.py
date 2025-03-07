import json
from pathlib import Path

# Define paths
feature_store_path = Path("data/features")
shuffled_words_file = feature_store_path / "shuffled_unique_words.json"
ratings_dir = Path("data/all_names")  # Directory containing the CSV files

# Define output files
output_files = {
    "trustworthy": feature_store_path / "word_ratings_trustworthy.json",
    "evil": feature_store_path / "word_ratings_evil.json",
    "smart": feature_store_path / "word_ratings_smart.json",
    "feminine": feature_store_path / "word_ratings_feminine.json"
}

# Load shuffled unique words
with open(shuffled_words_file, "r") as f:
    unique_words = json.load(f)

# Initialize dictionaries to store ratings
word_ratings = {
    "trustworthy": {},
    "evil": {},
    "smart": {},
    "feminine": {}
}

# Process each CSV file line by line
batch_size = 50000  # Define batch size for saving to avoid memory issues
current_batch = {"trustworthy": 0, "evil": 0, "smart": 0, "feminine": 0}

for file in ratings_dir.glob("*.csv"):
    print(f"Processing: {file.name}")
    
    category = None
    if "trustworthy" in file.name.lower():
        category = "trustworthy"
    elif "evil" in file.name.lower():
        category = "evil"
    elif "smart" in file.name.lower():
        category = "smart"
    elif "feminine" in file.name.lower():
        category = "feminine"
    
    if category is None:
        continue  # Skip files that don't match expected categories
    
    with file.open("r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue  # Skip malformed lines
            
            word, _, _, _, rating_value = parts
            rating_value = float(rating_value)
            
            if word not in word_ratings[category]:
                word_ratings[category][word] = []
            
            word_ratings[category][word].append(rating_value)
            
            # Save in batches to avoid memory overload
            current_batch[category] += 1
            if current_batch[category] >= batch_size:
                with open(output_files[category], "w") as f:
                    json.dump(word_ratings[category], f)
                print(f"✅ Batch saved for {category}")
                current_batch[category] = 0  # Reset batch counter

# Final save for remaining data
for category, output_file in output_files.items():
    with open(output_file, "w") as f:
        json.dump(word_ratings[category], f)
    print(f"✅ Final word ratings saved to {output_file}")
