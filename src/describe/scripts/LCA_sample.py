from pathlib import Path

import pandas as pd

###
# Gather the demographics for descriptive stats
processed_data_path = Path(
    "data",
    "processed_data",
)

LCA_path = Path(
    "data",
    "LCA",
)

core_data_path = Path(
    "data",
    "raw_data",
    "core",
)

general_info_path = Path(
    core_data_path,
    "abcd-general",
)

demographics_path = Path(
    general_info_path,
    "abcd_p_demo.csv",
)

demographics = pd.read_csv(
    demographics_path,
    index_col=0,
    low_memory=False,
)

abcd_y_lt_path = Path(
    general_info_path,
    "abcd_y_lt.csv",
)

abcd_y_lt = pd.read_csv(
    abcd_y_lt_path,
    index_col=0,
    low_memory=False,
)

abcd_y_lt_bl = abcd_y_lt[abcd_y_lt.eventname == "baseline_year_1_arm_1"]


cbcl_LCA_path = Path(
    LCA_path,
    "cbcl_class_member_prob.csv",
)

cbcl_LCA = pd.read_csv(
    cbcl_LCA_path,
    index_col=0,
    low_memory=False,
)


demographics_bl = demographics[demographics.eventname == "baseline_year_1_arm_1"]

family_income = demographics_bl["demo_comb_income_v2"].copy()

family_income = family_income.replace(777, 999)

# Select relevant demographic variables
# Select relevant demographic variables
des_vars = pd.DataFrame(
    {
        "demo_sex_v2": demographics_bl.demo_sex_v2,
        "race_ethnicity": demographics_bl.race_ethnicity,
        "family_income": family_income,
        "interview_age": abcd_y_lt_bl.interview_age,
    }
)

# Load CBCL dummy data
filtered_cbcl_save_path = Path("data", "LCA")
cbcl_dummy_path = Path(filtered_cbcl_save_path, "cbcl_t_no_mis_dummy.csv")
cbcl_dummy = pd.read_csv(cbcl_dummy_path, index_col=0, low_memory=False)

# Merge with demographic variables
cbcl_dummy_des_vars = cbcl_dummy.join(des_vars, how="left")

# Check for missing values
missing_values = cbcl_dummy_des_vars.isnull().sum()

# Calculate descriptive statistics (count and proportion) for each demographic variable
demographic_stats = {
    "demo_sex_v2": cbcl_dummy_des_vars["demo_sex_v2"]
    .value_counts()
    .to_frame(name="count"),
    "race_ethnicity": cbcl_dummy_des_vars["race_ethnicity"]
    .value_counts()
    .to_frame(name="count"),
    "family_income": cbcl_dummy_des_vars["family_income"]
    .value_counts()
    .to_frame(name="count"),
}

# Add proportion to each demographic variable's statistics
for key in demographic_stats:
    demographic_stats[key]["proportion"] = cbcl_dummy_des_vars[key].value_counts(
        normalize=True
    )

# Calculate mean and SD for interview age
interview_age_stats = (
    cbcl_dummy_des_vars["interview_age"]
    .agg(["mean", "std"])
    .to_frame(name="interview_age")
)

# Convert to a single DataFrame for easier viewing
demographic_stats_df = pd.concat(demographic_stats, axis=1)

# Display the results
print("Missing values in each column:\n", missing_values)
print("\nDescriptive Statistics (Count and Proportion):\n", demographic_stats_df)
print(
    "\nDescriptive Statistics for Interview Age (Mean and SD):\n", interview_age_stats
)


### Now get the descriptive statistics stratified by LCA class membership

cbcl_dummy_des_vars = cbcl_dummy_des_vars.join(cbcl_LCA.predicted_class, how="left")

# Initialize a dictionary to store the results for each class
class_stats = {}

for i in range(1, 5):
    cbcl_dummy_des_vars_class = cbcl_dummy_des_vars[
        cbcl_dummy_des_vars.predicted_class == i
    ]

    class_demographic_stats = {
        "demo_sex_v2": cbcl_dummy_des_vars_class["demo_sex_v2"]
        .value_counts()
        .to_frame(name="count"),
        "race_ethnicity": cbcl_dummy_des_vars_class["race_ethnicity"]
        .value_counts()
        .to_frame(name="count"),
        "family_income": cbcl_dummy_des_vars_class["family_income"]
        .value_counts()
        .to_frame(name="count"),
    }

    for key in class_demographic_stats:
        class_demographic_stats[key]["proportion"] = cbcl_dummy_des_vars_class[
            key
        ].value_counts(normalize=True)

    # Calculate mean and SD for interview age for the class
    class_interview_age_stats = (
        cbcl_dummy_des_vars_class["interview_age"]
        .agg(["mean", "std"])
        .to_frame(name="interview_age")
    )

    # Combine demographics and interview age stats
    class_stats[i] = pd.concat(
        [pd.concat(class_demographic_stats, axis=1), class_interview_age_stats.T]
    )

    print("Class", i)
    print(class_interview_age_stats.T)

# # Display the results for each class
# for i in range(1, 5):
#     print(
#         f"\nDescriptive Statistics for Predicted Class {i} (Count and Proportion):\n",
#         class_stats[i],
#     )
