from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

lca_path = Path(
    "data",
    "LCA",
)

# Load conditional probabilities for each latent class
cbcl_LCA_path = Path(
    lca_path,
    "lcmodel_prob_class.csv",
)

LCA_cond_prob = pd.read_csv(
    cbcl_LCA_path,
    index_col=0,
    low_memory=False,
)

LCA_cond_prob = LCA_cond_prob.rename(columns={"L2": "Diagnosis"})

LCA_cond_prob = LCA_cond_prob[LCA_cond_prob["Var2"] == "Pr(2)"]

LCA_cond_prob["Class"] = LCA_cond_prob["Var1"].str.extract(r"class (\d+):").astype(int)

syndrome_class_prob = LCA_cond_prob.pivot(
    index="Diagnosis",
    columns="Class",
    values="value",
)


# Plot a line chart
plt.figure(figsize=(14, 10))
for class_col in syndrome_class_prob.columns:
    plt.plot(
        syndrome_class_prob.index,
        syndrome_class_prob[class_col],
        marker="o",
        label=f"Class {class_col}",
    )

plt.title("")
plt.xlabel("Diagnosis")
plt.ylabel("Probability")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()  # Add a legend
plt.tight_layout()
plt.show()

# plt.savefig(
#     Path(
#         "data",
#         "plots",
#         "LCA",
#         "syndrome_class_prob.png",
#     )
# )
