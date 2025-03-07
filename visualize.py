import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load actual test ratings
test_data_path = Path("data/test")
feature_store_path = Path("data/features")

# Load actual test ratings (focusing on "evil" for now)
selected_trait = "evil"
test_file = test_data_path / f"word_ratings_{selected_trait}.json"

if test_file.exists():
    with open(test_file, "r", encoding="utf-8") as f:
        word_ratings_test = json.load(f)
else:
    raise FileNotFoundError(f"⚠️ Test file for {selected_trait} not found.")

# Load model predictions
predictions_file = feature_store_path / "word_predictions_test.json"
if predictions_file.exists():
    with open(predictions_file, "r", encoding="utf-8") as f:
        test_predictions = json.load(f)
else:
    raise FileNotFoundError("⚠️ No word predictions found. Run the model evaluation first.")

# Prepare data for visualization
data = []
if selected_trait in test_predictions:
    for word, ratings in word_ratings_test.items():
        if word in test_predictions[selected_trait]:
            predicted_value = test_predictions[selected_trait][word]
            
            # Compute statistics for actual values
            actual_median = np.median(ratings)
            actual_10th = np.percentile(ratings, 25)
            actual_90th = np.percentile(ratings, 75)

            data.append({
                "Word": word,
                "Actual_Median": actual_median,
                "Actual_10th": actual_10th,
                "Actual_90th": actual_90th,
                "Predicted": predicted_value
            })

# Convert to DataFrame
df = pd.DataFrame(data)

# Select 5 words for visualization
selected_words = df["Word"].unique()[:15]

# Create scatter plots for each selected word
for word in selected_words:
    word_data = df[df["Word"] == word]

    plt.figure(figsize=(6, 4))
    plt.scatter(word_data["Actual_Median"], word_data["Predicted"], color="red", label="Prediction")

    # Fixing errorbar shape issue
    yerr_lower = word_data["Actual_Median"].values - word_data["Actual_10th"].values
    yerr_upper = word_data["Actual_90th"].values - word_data["Actual_Median"].values
    yerr = np.array([yerr_lower, yerr_upper]).reshape(2, -1)  # Ensure proper shape

    plt.errorbar(
        word_data["Actual_Median"].values, word_data["Predicted"].values,
        yerr=yerr, fmt="o", color="blue", label="Actual Range (10%-90%)"
    )

    plt.axline((0, 0), slope=1, linestyle="dashed", color="gray")  # Ideal prediction line (y = x)
    plt.title(f"Prediction vs. Actual for '{word}' ({selected_trait})")
    plt.xlabel("Actual Median Rating")
    plt.ylabel("Predicted Rating")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid(True)
    plt.show()
