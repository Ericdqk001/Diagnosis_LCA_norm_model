import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Set global parameters to make all text bold
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


# Gather the demographics for descriptive stats
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

demographics_bl = demographics[demographics.eventname == "baseline_year_1_arm_1"]
family_income = demographics_bl["demo_comb_income_v2"].copy()
family_income = family_income.replace(777, 999)

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

data_splits_path = Path(processed_data_path, "data_splits.json")

with open(data_splits_path, "r") as f:
    data_splits = json.load(f)

feature_types = ["structural", "functional"]

# Plotting the distributions of demographics for each set
for modality in feature_types:
    modality_data_split = data_splits[modality]

    train_set_subs = modality_data_split["train"]
    val_set_subs = modality_data_split["val"]
    test_set_subs = modality_data_split["low_symptom_test"]

    sets = {
        "Train Set": train_set_subs,
        "Validation Set": val_set_subs,
        "Healthy Control (HC)": test_set_subs,
    }

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(
        f"Distribution of Demographics of each set for {modality.capitalize()} Modality",
        fontsize=16,
        fontweight="bold",
    )

    for i, (set_name, subs) in enumerate(sets.items()):
        subset = cbcl_dummy_des_vars.loc[subs]

        # Add row titles
        axes[i, 0].text(
            -0.5,
            0.5,
            set_name,
            va="center",
            ha="center",
            fontsize=14,
            fontweight="bold",
            rotation=90,
            transform=axes[i, 0].transAxes,
        )

        # Plot the distribution of sex
        sex_counts = subset["demo_sex_v2"].value_counts()
        sex_counts.plot(kind="bar", ax=axes[i, 0], color="skyblue")
        axes[i, 0].set_title("Sex", fontweight="bold")
        axes[i, 0].set_xlabel("Sex", fontweight="bold")
        axes[i, 0].set_ylabel("Count", fontweight="bold")
        axes[i, 0].set_xticklabels(
            sex_counts.index.map({1: "Male", 2: "Female", 3: "Intersex-Male"}),
            rotation=0,
            fontweight="bold",
        )

        # Plot the distribution of race/ethnicity
        race_counts = subset["race_ethnicity"].value_counts()
        race_counts.plot(kind="bar", ax=axes[i, 1], color="lightgreen")
        axes[i, 1].set_title("Race/Ethnicity", fontweight="bold")
        axes[i, 1].set_xlabel("Race/Ethnicity", fontweight="bold")
        axes[i, 1].set_ylabel("Count", fontweight="bold")
        axes[i, 1].set_xticklabels(
            race_counts.index.map(
                {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}
            ),
            rotation=0,
            fontweight="bold",
        )

        # Plot the distribution of family income
        income_counts = subset["family_income"].value_counts().sort_index()
        income_counts.plot(kind="bar", ax=axes[i, 2], color="lightcoral")
        axes[i, 2].set_title("Family Income", fontweight="bold")
        axes[i, 2].set_xlabel("Family Income", fontweight="bold")
        axes[i, 2].set_ylabel("Count", fontweight="bold")
        income_labels = [
            "Less than $5,000",
            "$5,000 - $11,999",
            "$12,000 - $15,999",
            "$16,000 - $24,999",
            "$25,000 - $34,999",
            "$35,000 - $49,999",
            "$50,000 - $74,999",
            "$75,000 - $99,999",
            "$100,000 - $199,999",
            "$200,000 and greater",
            "Not provided",
        ]
        axes[i, 2].set_xticks(range(len(income_counts)))
        axes[i, 2].set_xticklabels(income_labels, rotation=90, fontweight="bold")

        # Plot the distribution of interview age
        subset["interview_age"].plot(
            kind="hist", bins=20, ax=axes[i, 3], color="lightblue", edgecolor="black"
        )
        axes[i, 3].set_title("Interview Age", fontweight="bold")
        axes[i, 3].set_xlabel("Age in months", fontweight="bold")
        axes[i, 3].set_ylabel("Count", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
