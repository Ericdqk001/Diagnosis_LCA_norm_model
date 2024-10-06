# This script checks if the class membership had changed between using all samples with CBCL
# or samples with all data

from pathlib import Path

import pandas as pd

LCA_path = Path(
    "data",
    "LCA",
)

cbcl_LCA_path = Path(
    LCA_path,
    "cbcl_class_member_prob.csv",
)

cbcl_final_path = Path(
    LCA_path,
    "cbcl_final_class_member.csv",
)

cbcl_LCA = pd.read_csv(
    cbcl_LCA_path,
    index_col=0,
    low_memory=False,
)

final_sample_path = Path("data/processed_data/t1w_cortical_thickness_bl_pass.csv")

final_sample = pd.read_csv(
    final_sample_path,
    index_col=0,
    low_memory=False,
)

cbcl_LCA = cbcl_LCA.loc[final_sample.index]

cbcl_final = pd.read_csv(
    cbcl_final_path,
    index_col=0,
    low_memory=False,
)

# Filter by entropy <= 0.2
cbcl_LCA_filtered = cbcl_LCA[cbcl_LCA["entropy"] <= 0.2]
cbcl_final_filtered = cbcl_final[cbcl_final["entropy"] <= 0.2]

# Print the size of each class in cbcl_LCA_filtered
print("Class sizes in cbcl_LCA after entropy filtering:")
print(cbcl_LCA_filtered["predicted_class"].value_counts())

# Print the size of each class in cbcl_final_filtered
print("\nClass sizes in cbcl_final after entropy filtering:")
print(cbcl_final_filtered["predicted_class"].value_counts())
