import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Reload the processed data
processed_file = Path("data/processed/word_ratings.json")

if not processed_file.exists():
    raise FileNotFoundError(f"⚠️ Processed data file not found: {processed_file}")

# Read JSON file
with open(processed_file, "r", encoding="utf-8") as f:
    word_ratings = json.load(f)

# Convert to DataFrame
df = pd.DataFrame.from_dict(word_ratings, orient="index")

# Define output directory for plots
plot_output_dir = Path("plots")
plot_output_dir.mkdir(parents=True, exist_ok=True)

# Define exact figure size for consistency
fixed_size = (9, 5)

# Create and save plots for each trait (for pseudowords and company names)
for trait in ["trustworthy", "evil", "smart", "feminine"]:
    if trait in df.columns:
        fig, ax = plt.subplots(figsize=fixed_size)  # Ensure exact size
        sns.histplot(df[trait], bins=40, ax=ax)  # No KDE line
        ax.set_title(f"{trait.capitalize()} Ratings Distribution")
        ax.set_xlabel(f"{trait.capitalize()} Rating")
        ax.set_ylabel("Frequency")
        ax.grid(True)

        # Save plot
        plot_path = plot_output_dir / f"{trait}_ratings_distribution.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close plot to free memory

        print(f"✅ Saved plot for {trait} at {plot_path}")
