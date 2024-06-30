from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LCA_data_path = Path(
    "data",
    "LCA",
)

lca_class_prob_path = Path(
    LCA_data_path,
    "cbcl_class_member_prob.csv",
)

lca_class_prob = pd.read_csv(
    lca_class_prob_path,
    index_col=0,
    low_memory=False,
)

individual_entropy = lca_class_prob["entropy"]

# Select the rows where individual_entropy is higher than 0.20
selected_rows = lca_class_prob[lca_class_prob["entropy"] > 0.20]

# Get the corresponding subjects
subjects_with_high_entropy = selected_rows.index.tolist()

# Convert the list to a DataFrame
subjects_with_high_entropy_df = pd.DataFrame(
    subjects_with_high_entropy, columns=["subject"]
)

# Save the DataFrame to a csv file
# subjects_with_high_entropy_df.to_csv(
#     Path(
#         LCA_data_path,
#         "subjects_with_high_entropy.csv",
#     ),
#     index=False,
# )


# Plot the distribution of entropy
plt.figure(figsize=(10, 6))
sns.histplot(individual_entropy, bins=30, kde=True)
plt.title("Distribution of Individual-Level Entropy")
plt.xlabel("Entropy")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


####### Visualise the patterns of the excluded subjects


data_path = Path(
    "data",
    "raw_data",
    "core",
    "mental-health",
    "mh_p_cbcl.csv",
)

cbcl_t_vars_path = Path(
    "data",
    "var_dict",
    "cbcl_8_dim_t.csv",
)

cbcl = pd.read_csv(
    data_path,
    index_col=0,
    low_memory=False,
)

cbcl_t_vars_df = pd.read_csv(cbcl_t_vars_path)

cbcl_t_vars = cbcl_t_vars_df["var_name"].tolist()

baseline_cbcl = cbcl[cbcl["eventname"] == "baseline_year_1_arm_1"]

# Filter columns with t variables
filtered_cbcl = baseline_cbcl[cbcl_t_vars]



