import pandas as pd
import matplotlib.pyplot as plt

# Best performing models per trait with updated smartness result
data = {
    "Trait": ["Femininity", "Evilness", "Trustworthiness", "Smartness"],
    "Experiment": ["In-Domain", "Few-Shot", "Few-Shot", "In-Domain"],
    "Feature Set": ["Combined", "RobBERT", "Unigram", "Unigram"],
    "Test R²": [0.3109, 0.2226, 0.2508, 0.2757]  # Updated value for smartness
}

df = pd.DataFrame(data)

# Plotting
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
bars = ax.bar(df["Trait"], df["Test R²"], color=colors)

# Add experiment and feature info as labels
for bar, exp, feat in zip(bars, df["Experiment"], df["Feature Set"]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
            f"{exp}\n{feat}", ha='center', va='bottom', fontsize=8)

# Customizing plot
ax.set_title("Best Test R² per Trait Across All Models", fontsize=12)
ax.set_ylabel("Test R²")
ax.set_ylim(0, 0.35)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
# Save the figure
plt.savefig("best_models_per_trait.png", dpi=300, bbox_inches="tight")
plt.show()