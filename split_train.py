import json
import os
import random

# Directories
input_dir = "data/features/"
output_train_dir = "data/features/"  # Keep 85% here
output_test_dir = "data/test/"  # Move 15% here

# Ensure test directory exists
os.makedirs(output_test_dir, exist_ok=True)

# Trait JSON files
trait_files = [
    "word_ratings_evil.json",
    "word_ratings_trustworthy.json",
    "word_ratings_smart.json",
    "word_ratings_feminine.json",
]

# Split percentage for testing
test_size = 0.15

for file_name in trait_files:
    file_path = os.path.join(input_dir, file_name)

    # Load JSON data
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to list of (word, ratings) pairs
    words = list(data.keys())
    random.shuffle(words)

    # Select 15% of words for testing
    test_count = int(len(words) * test_size)
    test_words = words[:test_count]
    train_words = words[test_count:]

    # Create test and train dictionaries
    test_data = {word: data[word] for word in test_words}
    train_data = {word: data[word] for word in train_words}

    # Save updated training data
    with open(os.path.join(output_train_dir, file_name), "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)

    # Save new test data
    with open(os.path.join(output_test_dir, file_name), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    print(f"Processed {file_name}: {len(train_data)} training words, {len(test_data)} test words.")

print("✅ Data split complete. Test set stored in 'data/test/'.")
