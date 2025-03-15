import json
from pathlib import Path

# Define paths
ratings_dir = Path("data/all_names")  # Directory containing the CSV files
feature_store_path = Path("data/features")

# Define separate folders for training and test outputs
train_folder = feature_store_path / "train"
test_folder = feature_store_path / "test"

# Create folders if they don't exist
train_folder.mkdir(parents=True, exist_ok=True)
test_folder.mkdir(parents=True, exist_ok=True)

# Define output file names per trait for training and test
traits = ["trustworthy", "evil", "smart", "feminine"]
output_files_train = {trait: train_folder / f"word_ratings_train_{trait}.json" for trait in traits}
output_files_test  = {trait: test_folder / f"word_ratings_test_{trait}.json" for trait in traits}

# Initialize dictionaries to store ratings for training and test sets
word_ratings_train = {trait: {} for trait in traits}
word_ratings_test  = {trait: {} for trait in traits}

# Batch saving variables to avoid memory issues
batch_size = 200000
current_batch_train = {trait: 0 for trait in traits}
current_batch_test  = {trait: 0 for trait in traits}

# Process each CSV file in the ratings directory
for file in ratings_dir.glob("*.csv"):
    print(f"Processing: {file.name}")
    
    # Identify the trait category from the file name
    category = None
    file_lower = file.name.lower()
    for trait in traits:
        if trait in file_lower:
            category = trait
            break
    if category is None:
        continue  # Skip files that don't match expected categories

    with file.open("r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue  # Skip malformed lines

            word, _, word_type, _, rating_value = parts

            try:
                rating_value = float(rating_value)
            except ValueError:
                continue

            # Determine if the entry is for training or test based on word_type
            word_type_lower = word_type.lower()
            if word_type_lower in ["nepwoorden", "namen"]:
                # Training set: pseudowords and names
                target_dict = word_ratings_train[category]
                current_batch_train[category] += 1
            elif word_type_lower in ["bedrijfsnamen"]:
                # Test set: company names
                target_dict = word_ratings_test[category]
                current_batch_test[category] += 1
            else:
                continue  # Skip if word_type doesn't match

            # Append the rating for the word
            if word not in target_dict:
                target_dict[word] = []
            target_dict[word].append(rating_value)

            # Save in batches for training set
            if word_type_lower in ["nepwoorden", "namen"] and current_batch_train[category] >= batch_size:
                with open(output_files_train[category], "w", encoding="utf-8") as out_f:
                    json.dump(word_ratings_train[category], out_f)
                print(f"✅ Batch saved for training {category}")
                current_batch_train[category] = 0

            # Save in batches for test set
            if word_type_lower in ["bedrijfsnamen"] and current_batch_test[category] >= batch_size:
                with open(output_files_test[category], "w", encoding="utf-8") as out_f:
                    json.dump(word_ratings_test[category], out_f)
                print(f"✅ Batch saved for test {category}")
                current_batch_test[category] = 0

# Final save for any remaining data in training and test sets
for category, output_file in output_files_train.items():
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(word_ratings_train[category], out_f)
    print(f"✅ Final training ratings saved for {category} to {output_file}")

for category, output_file in output_files_test.items():
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(word_ratings_test[category], out_f)
    print(f"✅ Final test ratings saved for {category} to {output_file}")
