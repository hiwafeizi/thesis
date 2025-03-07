import polars as pl
import json
from pathlib import Path

# Define input and output directories
input_dir = Path("data/all_names")
output_file = Path("data/processed/word_ratings.json")

# Create output directory if it doesn't exist
output_file.parent.mkdir(parents=True, exist_ok=True)

# Traits to process
traits = ["trustworthy", "evil", "smart", "feminine"]

# Initialize dictionary for storing cumulative ratings
word_ratings = {}

# Process each CSV file
for trait in traits:
    file_path = input_dir / f"all_names_{trait}.csv"
    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}, skipping.")
        continue

    print(f"Processing {trait} data from {file_path}...")

    # Read the entire CSV file
    df = pl.read_csv(file_path)

    # Process data
    for row in df.iter_rows(named=True):
        word = row["word"]
        rating = row["rating_value"]

        if word not in word_ratings:
            word_ratings[word] = {}

        if trait not in word_ratings[word]:
            word_ratings[word][trait] = 0.0

        word_ratings[word][trait] += rating

    # Normalize by dividing each summed value by 1000
    for word in word_ratings:
        if trait in word_ratings[word]:  
            word_ratings[word][trait] /= 1000

    print(f"✅ Processed data for {trait}")

# Save final JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(word_ratings, f, indent=4)

print(f"✅ Final dataset saved at {output_file}")
