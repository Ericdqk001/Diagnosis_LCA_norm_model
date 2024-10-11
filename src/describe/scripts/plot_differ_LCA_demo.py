import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style for professional appearance
plt.style.use("ggplot")


def plot_demo_difference(original_class_df, retained_subs, des_vars, low_entropy):
    sns.set(
        style="whitegrid"
    )  # Setting the seaborn style to whitegrid for a clean layout

    # Extract indices for each class
    original_inter = original_class_df[original_class_df["predicted_class"] == 2].index
    original_exter = original_class_df[original_class_df["predicted_class"] == 3].index
    original_dysregulate = original_class_df[
        original_class_df["predicted_class"] == 4
    ].index

    # Extract retained subjects for each test condition
    retained_inter = retained_subs["internalising_test"]
    retained_exter = retained_subs["externalising_test"]
    retained_dysregulate = retained_subs["high_symptom_test"]

    # Exclude high entropy subjects if requested
    if low_entropy:
        high_entropy_subs = original_class_df[original_class_df["entropy"] > 0.2].index
        original_inter = [sub for sub in original_inter if sub not in high_entropy_subs]
        original_exter = [sub for sub in original_exter if sub not in high_entropy_subs]
        original_dysregulate = [
            sub for sub in original_dysregulate if sub not in high_entropy_subs
        ]
        retained_inter = [sub for sub in retained_inter if sub not in high_entropy_subs]
        retained_exter = [sub for sub in retained_exter if sub not in high_entropy_subs]
        retained_dysregulate = [
            sub for sub in retained_dysregulate if sub not in high_entropy_subs
        ]

    # Define class sets
    sets = {
        "Predominantly Internalizing": (original_inter, retained_inter),
        "Predominantly Externalizing": (original_exter, retained_exter),
        "Highly Dysregulated": (original_dysregulate, retained_dysregulate),
    }

    # Creating subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(
        "Demographic Differences Between Original and Retained Samples",
        fontsize=16,
        fontweight="bold",
    )

    # Iterating through each class set
    for i, (set_name, (original_subs, retained_subs)) in enumerate(sets.items()):
        original_subset = des_vars.loc[original_subs]
        retained_subset = des_vars.loc[retained_subs]

        # Row titles
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

        # Plotting sex distribution
        plot_distribution_comparison(
            "demo_sex_v2", "Sex", original_subset, retained_subset, axes[i, 0]
        )

        # Plotting race/ethnicity distribution
        plot_distribution_comparison(
            "race_ethnicity",
            "Race/Ethnicity",
            original_subset,
            retained_subset,
            axes[i, 1],
        )

        # Plotting family income distribution
        plot_distribution_comparison(
            "family_income",
            "Family Income",
            original_subset,
            retained_subset,
            axes[i, 2],
        )

        # Plotting interview age distribution
        plot_age_distribution(
            "Interview Age", original_subset, retained_subset, axes[i, 3]
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_distribution_comparison(column, title, original_subset, retained_subset, ax):
    original_counts = original_subset[column].value_counts()
    retained_counts = retained_subset[column].value_counts()
    df = pd.DataFrame(
        {"Original": original_counts, "Retained": retained_counts}
    ).fillna(0)
    df.plot(kind="bar", ax=ax, color=["skyblue", "orange"])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(title, fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_xticklabels(df.index.map(str), rotation=0, fontweight="bold")


def plot_age_distribution(title, original_subset, retained_subset, ax):
    original_subset["interview_age"].plot(
        kind="hist",
        bins=20,
        ax=ax,
        color="skyblue",
        edgecolor="black",
        alpha=0.5,
        label="Original",
    )
    retained_subset["interview_age"].plot(
        kind="hist",
        bins=20,
        ax=ax,
        color="orange",
        edgecolor="black",
        alpha=0.5,
        label="Retained",
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Age in months", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.legend()


# Main block where data is loaded and function is called
if __name__ == "__main__":
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

    des_vars = pd.DataFrame(
        {
            "demo_sex_v2": demographics_bl.demo_sex_v2,
            "race_ethnicity": demographics_bl.race_ethnicity,
            "family_income": family_income,
            "interview_age": abcd_y_lt_bl.interview_age,
        }
    )

    lca_path = Path("data", "LCA")
    lca_class_memberships_path = Path(lca_path, "cbcl_class_member_prob.csv")
    lca_class_memberships = pd.read_csv(
        lca_class_memberships_path, index_col=0, low_memory=False
    )
    class_ids_to_plot = [2, 3, 4]
    original_class_df = lca_class_memberships[
        lca_class_memberships["predicted_class"].isin(class_ids_to_plot)
    ]

    data_splits_path = Path(processed_data_path, "data_splits.json")
    with open(data_splits_path, "r") as f:
        data_splits = json.load(f)
        retained_subs = data_splits["structural"]

    plot_demo_difference(original_class_df, retained_subs, des_vars, low_entropy=True)
