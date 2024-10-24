# %%
import json
from pathlib import Path

import pandas as pd

# Define mappings based on provided labels
sex_mapping = {
    1: "Male",
    2: "Female",
}

race_ethnicity_mapping = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}

income_mapping = {
    1: "Less than $5,000",
    2: "$5,000 through $11,999",
    3: "$12,000 through $15,999",
    4: "$16,000 through $24,999",
    5: "$25,000 through $34,999",
    6: "$35,000 through $49,999",
    7: "$50,000 through $74,999",
    8: "$75,000 through $99,999",
    9: "$100,000 through $199,999",
    10: "$200,000 and greater",
    999: "Not Provided",
}

# Paths
processed_data_path = Path("data", "processed_data")
LCA_path = Path("data", "LCA")
core_data_path = Path("data", "raw_data", "core")
general_info_path = Path(core_data_path, "abcd-general")

demographics_path = Path(general_info_path, "abcd_p_demo.csv")
demographics = pd.read_csv(demographics_path, index_col=0, low_memory=False)

abcd_y_lt_path = Path(general_info_path, "abcd_y_lt.csv")
abcd_y_lt = pd.read_csv(abcd_y_lt_path, index_col=0, low_memory=False)
abcd_y_lt_bl = abcd_y_lt[abcd_y_lt.eventname == "baseline_year_1_arm_1"]

cbcl_LCA_path = Path(LCA_path, "cbcl_class_member_prob.csv")
cbcl_LCA = pd.read_csv(cbcl_LCA_path, index_col=0, low_memory=False)


# Read the data_splits.json to get 'HCs' cohort
data_splits_path = Path("data", "processed_data", "data_splits.json")
with open(data_splits_path, "r") as f:
    data_splits = json.load(f)


demographics_bl = demographics[demographics.eventname == "baseline_year_1_arm_1"]
family_income = demographics_bl["demo_comb_income_v2"].replace(777, 999)

# Select relevant demographic variables
des_vars = pd.DataFrame(
    {
        "demo_sex_v2": demographics_bl.demo_sex_v2,
        "race_ethnicity": demographics_bl.race_ethnicity,
        "family_income": family_income,
        "interview_age": abcd_y_lt_bl.interview_age,
    }
)

# Merge with demographic variables
cbcl_LCA = cbcl_LCA.join(
    des_vars,
    how="left",
)

# Filter for low-entropy classes
low_entropy = False
if low_entropy:
    cbcl_LCA = cbcl_LCA[cbcl_LCA["entropy"] <= 0.2]

# Apply mappings to demographic columns
cbcl_LCA["demo_sex_v2"] = cbcl_LCA["demo_sex_v2"].map(sex_mapping)
cbcl_LCA["race_ethnicity"] = cbcl_LCA["race_ethnicity"].map(race_ethnicity_mapping)
cbcl_LCA["family_income"] = cbcl_LCA["family_income"].map(income_mapping)


# Define the demographic summary function
def summarize_demographics(df):
    # Total number of participants
    total = len(df)

    # Calculate n (%) for each demographic variable
    summary = {
        "N": total,
        "Sex (Male)": f"{(df['demo_sex_v2'] == 'Male').sum()} ({(df['demo_sex_v2'] == 'Male').mean() * 100:.1f}%)",
        "Sex (Female)": f"{(df['demo_sex_v2'] == 'Female').sum()} ({(df['demo_sex_v2'] == 'Female').mean() * 100:.1f}%)",
        "Mean Age": df["interview_age"].mean(),
        "Age Std Dev": df["interview_age"].std(),
        "< 5,000": f"{(df['family_income'] == 'Less than $5,000').sum()} ({(df['family_income'] == 'Less than $5,000').mean() * 100:.1f}%)",
        "5,000 - 11,999": f"{(df['family_income'] == '$5,000 through $11,999').sum()} ({(df['family_income'] == '$5,000 through $11,999').mean() * 100:.1f}%)",
        "Income $12,000-$15,999": f"{(df['family_income'] == '$12,000 through $15,999').sum()} ({(df['family_income'] == '$12,000 through $15,999').mean() * 100:.1f}%)",
        "16,000 - 24,999": f"{(df['family_income'] == '$16,000 through $24,999').sum()} ({(df['family_income'] == '$16,000 through $24,999').mean() * 100:.1f}%)",
        "25,000 - 34,999": f"{(df['family_income'] == '$25,000 through $34,999').sum()} ({(df['family_income'] == '$25,000 through $34,999').mean() * 100:.1f}%)",
        "35,000 - 49,999": f"{(df['family_income'] == '$35,000 through $49,999').sum()} ({(df['family_income'] == '$35,000 through $49,999').mean() * 100:.1f}%)",
        "50,000 - 74,999": f"{(df['family_income'] == '$50,000 through $74,999').sum()} ({(df['family_income'] == '$50,000 through $74,999').mean() * 100:.1f}%)",
        "75,000 - 99,999": f"{(df['family_income'] == '$75,000 through $99,999').sum()} ({(df['family_income'] == '$75,000 through $99,999').mean() * 100:.1f}%)",
        "100,000 - 199,999": f"{(df['family_income'] == '$100,000 through $199,999').sum()} ({(df['family_income'] == '$100,000 through $199,999').mean() * 100:.1f}%)",
        "> 200,000": f"{(df['family_income'] == '$200,000 and greater').sum()} ({(df['family_income'] == '$200,000 and greater').mean() * 100:.1f}%)",
        "Income Not Provided": f"{(df['family_income'] == 'Not Provided').sum()} ({(df['family_income'] == 'Not Provided').mean() * 100:.1f}%)",
        "White": f"{(df['race_ethnicity'] == 'White').sum()} ({(df['race_ethnicity'] == 'White').mean() * 100:.1f}%)",
        "Black": f"{(df['race_ethnicity'] == 'Black').sum()} ({(df['race_ethnicity'] == 'Black').mean() * 100:.1f}%)",
        "Hispanic": f"{(df['race_ethnicity'] == 'Hispanic').sum()} ({(df['race_ethnicity'] == 'Hispanic').mean() * 100:.1f}%)",
        "Asian": f"{(df['race_ethnicity'] == 'Asian').sum()} ({(df['race_ethnicity'] == 'Asian').mean() * 100:.1f}%)",
        "Other": f"{(df['race_ethnicity'] == 'Other').sum()} ({(df['race_ethnicity'] == 'Other').mean() * 100:.1f}%)",
    }

    return pd.Series(summary)


# Create a dictionary to hold the summaries for each class
class_summaries = {}

# Loop through the predicted classes and calculate summaries
for predicted_class in range(1, 5):
    class_data = cbcl_LCA[cbcl_LCA["predicted_class"] == predicted_class]
    class_summaries[f"Class {predicted_class}"] = summarize_demographics(class_data)


# Combine the summaries into a single DataFrame
demographic_summary_df = pd.DataFrame(class_summaries)

# Export to Excel
demographic_summary_df.to_excel(
    "original_lca_demographic_summary.xlsx", engine="openpyxl"
)

# Display the dataframe
print(demographic_summary_df)
