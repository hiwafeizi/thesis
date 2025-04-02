import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # for regression and correlation calculation
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

# Define the reference trait
reference_trait = "feminine"

# List of other traits to compare with "feminine"
other_traits = [trait for trait in ["trustworthy", "evil", "smart"] if trait in df.columns]

# Create scatter plots to examine the relationship between "feminine" and each of the other traits
for trait in other_traits:
    if reference_trait in df.columns:
        fig, ax = plt.subplots(figsize=fixed_size)
        
        # Scatter plot: x-axis for 'feminine', y-axis for the other trait
        ax.scatter(df[reference_trait], df[trait], alpha=0.6)
        
        # Fit and plot a regression line using numpy.polyfit
        slope, intercept = np.polyfit(df[reference_trait], df[trait], 1)
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, linestyle='--', color='red', alpha=0.7)
        
        # Calculate the Pearson correlation coefficient
        corr_coef = np.corrcoef(df[reference_trait], df[trait])[0, 1]
        
        # Set title and labels
        ax.set_title(f"Relationship between {reference_trait.capitalize()} and {trait.capitalize()} (r = {corr_coef:.2f})")
        ax.set_xlabel(f"{reference_trait.capitalize()} Rating")
        ax.set_ylabel(f"{trait.capitalize()} Rating")
        ax.grid(True)
        
        # Save the plot
        plot_path = plot_output_dir / f"{reference_trait}_vs_{trait}_scatter.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close plot to free memory
        
        print(f"✅ Saved scatter plot for {reference_trait} vs {trait} at {plot_path}")
