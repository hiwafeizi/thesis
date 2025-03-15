import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Reload the processed data
processed_file = Path("data/processed/company_ratings.json")
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
fixed_size = (11, 5)

# Create and save plots for each trait (for pseudowords and company names)
for trait in ["trustworthy", "evil", "smart", "feminine"]:
    if trait in df.columns:
        fig, ax = plt.subplots(figsize=fixed_size)  # Ensure exact size
        sns.histplot(df[trait], bins=40, ax=ax)  # No KDE line
        ax.set_title(f"{trait.capitalize()} Ratings Distribution of Company names (mean of 1000 variations)")
        ax.set_xlabel(f"{trait.capitalize()} Rating")
        ax.set_ylabel("Frequency")
        ax.grid(True)

        # Calculate the lowest, highest, and three random words for the trait
        lowest_word = df[trait].idxmin()
        highest_word = df[trait].idxmax()
        lowest_value = df.loc[lowest_word, trait]
        highest_value = df.loc[highest_word, trait]
        
        # Exclude lowest and highest words from random selection
        remaining_words = [w for w in df.index if w not in [lowest_word, highest_word]]
        if len(remaining_words) >= 3:
            random_words = random.sample(remaining_words, 3)
        else:
            random_words = remaining_words

        # Create a list with tuples: (word, rating_value, type)
        words_info = []
        words_info.append((lowest_word, lowest_value, "Lowest"))
        words_info.append((highest_word, highest_value, "Highest"))
        for word in random_words:
            words_info.append((word, df.loc[word, trait], "Random"))
        
        # Sort the words by rating value so vertical lines are drawn in order
        words_info.sort(key=lambda x: x[1])
        
        # Draw vertical dashed red lines for each word at its rating value
        for word, value, _ in words_info:
            ax.axvline(x=value, linestyle='--', color='red', alpha=0.7)
        
        # Construct annotation text for the top frame
        annotation_lines = []
        for word, value, category in words_info:
            annotation_lines.append(f"{category}: {word} ({value:.2f})")
        annotation_text = "\n".join(annotation_lines)
        
        # Place the annotation in a frame at the top of the plot (centered horizontally)
        ax.text(0.5, 0.95, annotation_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        
        # Save plot
        plot_path = plot_output_dir / f"{trait}_ratings_distribution.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close plot to free memory
 
        print(f"✅ Saved plot for {trait} at {plot_path}")
