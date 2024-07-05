# TODO plot the sample size of each site with predicted class distribution
# TODO: Test the effectiveness of neuroCombat in removing site effects
# TODO: Test the site effect on intracranial volume (Very Strong)
# TODO: Test the site effect on predicted class membership

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


def describe_site():
    process_data_path = Path(
        "data",
        "processed_data",
    )

    all_brain_features_precon_path = Path(
        process_data_path,
        "all_brain_features_precon.csv",
    )

    all_brain_features_precon = pd.read_csv(
        all_brain_features_precon_path,
        index_col=0,
        low_memory=False,
    )

    all_brain_features_path = Path(
        process_data_path,
        "all_brain_features_resid_exc_sex.csv",
    )

    all_brain_features_postcon = pd.read_csv(
        all_brain_features_path,
        index_col=0,
        low_memory=False,
    )

    # Plot the sample size of each site by sex
    plot_sex_proportion(all_brain_features_precon)

    # Plot the sample size of each site by predicted class memberships
    plot_class_proportion(all_brain_features_precon)

    # Plot the sample size of each site by income
    plot_income_proportion(all_brain_features_precon)

    # Statistically test the effect of site on predicted class membership
    site_on_latent_class(all_brain_features_precon)

    # Plot the effect of site on brain features
    plot_site_effect_on_brain_features(all_brain_features_precon)

    plot_site_effect_on_brain_features(all_brain_features_postcon)


def plot_sex_proportion(df):
    # Calculate counts for each sex within each site
    sex_counts = (
        df.groupby(
            ["label_site", "demo_sex_v2"],
        )
        .size()
        .unstack(fill_value=0)
    )

    # Calculate total counts by site
    sex_counts["Total"] = sex_counts.sum(axis=1)

    # Normalize to get proportions for colors
    proportions = sex_counts.loc[:, sex_counts.columns != "Total"].div(
        sex_counts["Total"], axis=0
    )

    # Adjust figure size dynamically based on the number of sites
    num_sites = len(sex_counts)
    fig_width = max(10, num_sites * 0.5)
    fig_height = 6

    print(fig_width, fig_height)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create a stacked bar plot
    proportions.plot(
        kind="bar",
        stacked=True,
        color=["#1f77b4", "#ff7f0e"],
        ax=ax,
    )
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Site")

    # Overlay total counts as bar labels
    for i, total in enumerate(sex_counts["Total"]):
        ax.text(
            i,
            1,
            str(total),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.title("Sample Size and Sex Proportion by Site")
    plt.legend(title="Sex", labels=["Male", "Female"])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_class_proportion(df):
    # Map class labels to descriptive names
    class_labels = {
        1: "Low Symptom",
        2: "Predominantly Internalising",
        3: "Predominantly Externalising",
        4: "Highly Dysregulated",
    }

    # Replace numeric class labels with descriptive names
    df["predicted_class"] = df["predicted_class"].map(class_labels)

    # Calculate counts for each predicted class within each site
    class_counts = (
        df.groupby(["label_site", "predicted_class"]).size().unstack(fill_value=0)
    )

    # Calculate proportions for each class within each site
    class_proportions = class_counts.div(class_counts.sum(axis=1), axis=0)

    # Calculate total counts for each site
    total_counts = class_counts.sum(axis=1)

    # Plot the data
    fig, ax = plt.subplots(figsize=(14.5, 6))

    # Plot bars for each site with proportions as stacked bars
    class_proportions.plot(kind="bar", stacked=True, ax=ax)

    # Overlay total counts as bar labels
    for i, total in enumerate(total_counts):
        ax.text(
            i,
            1,
            str(total),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Set plot labels and title
    ax.set_xlabel("Site")
    ax.set_ylabel("Proportion")
    ax.set_title("Proportion of Predicted Classes in Each Site")

    # Create a legend
    ax.legend(title="Predicted Class")

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_income_proportion(df):
    # Ensure that the necessary columns are present in the dataframe
    required_columns = ["label_site", "demo_comb_income_v2"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the dataframe")

    # Calculate counts for each income category within each site
    income_counts = (
        df.groupby(["label_site", "demo_comb_income_v2"]).size().unstack(fill_value=0)
    )

    # Calculate proportions for each income category within each site
    income_proportions = income_counts.div(income_counts.sum(axis=1), axis=0)

    # Plot the data
    fig, ax = plt.subplots(figsize=(14.5, 6))

    # Plot bars for each site with proportions as stacked bars
    income_proportions.plot(kind="bar", stacked=True, ax=ax)

    # Set plot labels and title
    ax.set_xlabel("Site")
    ax.set_ylabel("Proportion")
    ax.set_title("Proportion of Income Categories in Each Site")

    # Create a legend
    ax.legend(title="Income Category")

    # Display the plot
    plt.tight_layout()
    plt.show()


def site_on_latent_class(df):

    predicted_class = df["predicted_class"]

    label_site = df["label_site"]

    # Create a contingency table
    contingency_table = pd.crosstab(predicted_class, label_site)

    # Perform the Chi-Square test of independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Print the results
    print("Chi-Square Test Statistic:", chi2)
    print("P-Value:", p)
    print("Degrees of Freedom:", dof)
    print("Expected Frequencies:\n", expected)

    # Interpret the result
    if p < 0.05:
        print(
            "There is a significant relationship between predicted_class and label_site (reject H0)"
        )
    else:
        print(
            "There is no significant relationship between predicted_class and label_site (fail to reject H0)"
        )


def plot_site_effect_on_brain_features(df, feature="mrisdp_1"):
    # Ensure that the necessary columns are present in the dataframe
    required_columns = ["label_site", feature]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the dataframe")

    # Set the size of the plot
    plt.figure(figsize=(14.5, 6))

    # Create a boxplot to visualize the distribution of the feature across sites
    sns.boxplot(x="label_site", y=feature, data=df)

    # Set plot labels and title
    plt.xlabel("Site")
    plt.ylabel("Feature Value")
    plt.title(f"Distribution of {feature} Across Sites")

    # Rotate x-axis labels for better readability if there are many sites
    plt.xticks(rotation=45)

    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    describe_site()
