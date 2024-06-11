import matplotlib.pyplot as plt
import pandas as pd

# Load the data
file_path = "data/LCA/lcmodel_prob_class.csv"  # Replace with your actual file path
LCA_cond_prob = pd.read_csv(file_path, index_col=0, low_memory=False)

# Define mappings
syndrome_map = {
    "Anxiety/Depress": "cbcl_scr_syn_anxdep_t",
    "Withdraw/Depress": "cbcl_scr_syn_withdep_t",
    "Somatic": "cbcl_scr_syn_somatic_t",
    "Social": "cbcl_scr_syn_social_t",
    "Thought": "cbcl_scr_syn_thought_t",
    "Attention": "cbcl_scr_syn_attention_t",
    "RuleBreak": "cbcl_scr_syn_rulebreak_t",
    "Aggressive": "cbcl_scr_syn_aggressive_t",
}

inverse_syndrome_map = {v: k for k, v in syndrome_map.items()}

# Process data
LCA_cond_prob = LCA_cond_prob.rename(columns={"L2": "Syndrome"})
LCA_cond_prob = LCA_cond_prob[LCA_cond_prob["Var2"] == "Pr(2)"]
LCA_cond_prob["Syndrome"] = LCA_cond_prob["Syndrome"].map(inverse_syndrome_map)
LCA_cond_prob["Class"] = LCA_cond_prob["Var1"].str.extract(r"class (\d+):").astype(int)

syndrome_class_prob = LCA_cond_prob.pivot(
    index="Syndrome", columns="Class", values="value"
)

# Reorder for better visualization
new_order = [
    "Aggressive",
    "RuleBreak",
    "Attention",
    "Thought",
    "Anxiety/Depress",
    "Withdraw/Depress",
    "Somatic",
    "Social",
]
syndrome_class_prob = syndrome_class_prob.reindex(new_order)


# Rename columns


# Plotting


plt.figure(figsize=(14, 10), facecolor="white")
ax = plt.gca()

bright_colors = [
    "#FF5733",
    "#33FF57",
    "#3357FF",
    "#FF33A1",
    "#FFDB33",
    "#33FFF0",
    "#FF3333",
    "#8D33FF",
]

class_names = [
    "Low Symptom",
    "Predominantly Internalising",
    "Predominantly Externalising",
    "Highly Dysregulated",
]

plt.figure(figsize=(10, 6.7))

for idx, class_col in enumerate(syndrome_class_prob.columns):
    plt.plot(
        syndrome_class_prob.index,
        syndrome_class_prob[class_col],
        marker="o",
        markersize=10,
        markerfacecolor="none",  # Make the marker an empty circle
        markeredgewidth=2,
        markeredgecolor=bright_colors[
            idx % len(bright_colors)
        ],  # Add this line to color the marker edge
        color=bright_colors[idx % len(bright_colors)],
        label=class_names[idx],
        linewidth=4,  # Increase the line width
    )

plt.title("Syndrome Class Probability", color="black", fontweight="bold")
plt.xlabel("", color="black", fontweight="bold")
plt.ylabel("Probability", color="black", fontweight="bold")
plt.xticks(rotation=45, color="black", fontweight="bold")
plt.yticks(color="black", fontweight="bold")
plt.legend(prop={"weight": "bold"})  # Set legend labels to bold
plt.grid(False)  # This line removes the grid
ax.set_facecolor("white")  # This sets the plot background color to white
plt.tight_layout()
plt.show()
