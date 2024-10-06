import matplotlib.pyplot as plt
import pandas as pd

# Load the data
file_path = "data/LCA/lcmodel_prob_class_final_sample.csv"  # Replace with your actual file path
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

# Plotting
plt.figure(figsize=(10, 6.7))
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
    "Class 1 (88.36%)",
    "Class 2 (7.13%)",
    "Class 3 (2.46%)",
    "Class 4 (2.04%)",
]

plt.figure(figsize=(10, 6.7))

for idx, class_col in enumerate(syndrome_class_prob.columns):
    plt.plot(
        syndrome_class_prob.index,
        syndrome_class_prob[class_col],
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markeredgecolor=bright_colors[idx % len(bright_colors)],
        color=bright_colors[idx % len(bright_colors)],
        label=class_names[idx],
        linewidth=2,
    )

plt.title("Syndrome Class Probability", fontsize=14, fontweight="bold")
plt.xlabel("", fontsize=12, fontweight="bold")
plt.ylabel("Probability", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, fontsize=10, fontweight="bold")
plt.yticks(fontsize=10, fontweight="bold")
plt.legend(prop={"weight": "bold", "size": 10}, frameon=False)
plt.grid(False)
ax.set_facecolor("white")
plt.tight_layout()
plt.show()
