import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load JSON data
json_path = Path("data/processed/company_ratings.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Create DataFrame from JSON data
df = pd.DataFrame.from_dict(data, orient="index")

# Compute overall averages for each trait
overall_avg = df.mean()

# Compute averages for words containing at least one 'x'
words_with_x = df.index[df.index.str.lower().str.contains('x')]
avg_with_x = df.loc[words_with_x].mean()

# Compute averages for words containing at least one 'a'
words_with_a = df.index[df.index.str.lower().str.contains('la')]
avg_with_la = df.loc[words_with_a].mean()

# Compute averages for words ending with 'a' (the "a." condition)
words_ending_a = df.index[df.index.str.lower().str.endswith('a')]
avg_ending_a = df.loc[words_ending_a].mean()

# Create a summary table with the computed averages (rounded to 2 decimals)
results_table = pd.DataFrame({
    'Overall': overall_avg,
    "Contains 'x'": avg_with_x,
    "Contains 'la'": avg_with_la,
    "Ends with 'a'": avg_ending_a
}).round(2)

# Create a plot to display the table
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=results_table.values,
                 colLabels=results_table.columns,
                 rowLabels=results_table.index,
                 cellLoc='center',
                 loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Customize cell styling
for key, cell in table.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_linewidth(1)
    # Header cells: column headers (row index -1) and row labels (column index -1)
    if key[0] == -1 or key[1] == -1:
        cell.set_facecolor('#40466e')
        cell.set_text_props(color='w', weight='bold')
    else:
        cell.set_facecolor('#f1f1f2')

plt.title("Average Ratings by Condition", fontsize=12)

# Save the table plot
output_path = Path("plots/average_ratings_table.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Table plot saved to {output_path}")
