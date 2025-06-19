import pandas as pd
import matplotlib.pyplot as plt

# Best performing models per trait with updated smartness result
data = {
    "Trait": ["Femininity", "Evilness", "Trustworthiness", "Smartness"],
    "Test R²": [0.3109, 0.2226, 0.2508, 0.2757]
}

df = pd.DataFrame(data)

# Plotting
fig, ax = plt.subplots(figsize=(6, 3))
bars = ax.bar(df["Trait"], df["Test R²"], color="#6baed6")

# Add R² values on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
            f"{height:.2f}", ha='center', va='bottom', fontsize=9)

# Style: minimal axes
ax.set_ylabel("Test R²")
ax.set_ylim(0, 0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("lightgray")
ax.tick_params(axis='y', length=0)
ax.tick_params(axis='x', length=0)

plt.tight_layout()
plt.savefig("best_r2_barplot.png", dpi=300, bbox_inches="tight")
plt.show()
