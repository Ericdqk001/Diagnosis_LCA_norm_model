# %%
### Swap the class memberships such that the 'low symptom' class is the first class
from pathlib import Path

import pandas as pd

LCA_path = Path(
    "data",
    "LCA",
)
# %%
# Load cbcl-LCA data
# Swap the class 1 with 2 as well for the cbcl_class_member_prob.csv file
# (produced from the lca_cbcl.R script)
cbcl_LCA_path = Path(
    LCA_path,
    "cbcl_final_class_member.csv",
)

cbcl_LCA = pd.read_csv(
    cbcl_LCA_path,
    index_col=0,
    low_memory=False,
)

# Swap values in the 'predicted_class' column
cbcl_LCA["predicted_class"].replace(
    {1: 3, 3: 1},
    inplace=True,
)

cbcl_LCA["predicted_class"].astype(int)

# Swap column names between "ClassProb_1" and "ClassProb_2"
temp_column = cbcl_LCA["ClassProb_1"].copy()

cbcl_LCA = cbcl_LCA.drop("ClassProb_1", axis=1)

cbcl_LCA.rename(columns={"ClassProb_3": "ClassProb_1"}, inplace=True)

cbcl_LCA["ClassProb_3"] = temp_column

cbcl_LCA.to_csv(
    cbcl_LCA_path,
    index=True,
)

# %%
