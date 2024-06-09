import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set global parameters to make all text bold
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


def plot_histograms(
    feature: str,
    output_data: pd.DataFrame,
    p_values_df: pd.DataFrame,
    metric: str = "mahalanobis_distance",
):
    # Extract distances for each group
    control_distance = output_data[metric][
        output_data["low_symp_test_subs"] == 1
    ].values
    internalising_distance = output_data[metric][
        output_data["inter_test_subs"] == 1
    ].values
    exter_test_distance = output_data[metric][
        output_data["exter_test_subs"] == 1
    ].values
    high_test_distance = output_data[metric][output_data["high_test_subs"] == 1].values

    # Create a figure with subplots (1 row, 3 columns)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    fig.suptitle(
        "Distribution of Deviations by Group for Feature: {}".format(feature),
        fontweight="bold",
    )

    # Custom color palette
    colors = ["#ff7f0e", "#2ca02c", "#d62728"]  # More distinctive color palette
    group_names = [
        "Predominantly Internalising",
        "Predominantly Externalising",
        "Highly Dysregulated",
    ]

    # Labels and distances for each plot
    groups = zip(
        group_names, [internalising_distance, exter_test_distance, high_test_distance]
    )

    p_value_cohort_map = {
        "Predominantly Internalising": "inter_test",
        "Predominantly Externalising": "exter_test",
        "Highly Dysregulated": "high_test",
    }

    # Plot histograms
    for ax, (group_name, group_distance), color in zip(axes, groups, colors):
        ax.hist(
            control_distance,
            bins=20,
            alpha=0.5,
            color="grey",
            label="Control",
            histtype="stepfilled",
        )
        ax.hist(
            group_distance,
            bins=20,
            alpha=0.75,
            color=color,
            label="Case",
            histtype="stepfilled",
        )
        ax.set_title("{} vs Control".format(group_name))

        # Add vertical lines for the median of each distribution
        ax.axvline(
            np.median(control_distance),
            color="grey",
            linestyle="--",
            label="Control Median",
        )
        ax.axvline(
            np.median(group_distance),
            color=color,
            linestyle="--",
            label="Case Median",
        )

        # Fetch and annotate the p-value for the current group
        p_value = p_values_df.loc[
            p_values_df["Cohort"] == p_value_cohort_map[group_name], "P_value"
        ].values[0]
        ax.text(
            0.95,
            0.95,
            f"p-value: {p_value:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor="black",
                facecolor="white",
                alpha=0.8,
            ),
        )

        ax.set_xlabel("Deviation Z-Score")

        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_boxplots(
    feature: str,
    output_data: pd.DataFrame,
    metric: str = "mahalanobis_distance",
):
    # Extract Mahalanobis distances for each group
    control_distance = output_data[metric][
        output_data["low_symp_test_subs"] == 1
    ].values
    inter_test_distance = output_data[metric][
        output_data["inter_test_subs"] == 1
    ].values
    exter_test_distance = output_data[metric][
        output_data["exter_test_subs"] == 1
    ].values
    high_test_distance = output_data[metric][output_data["high_test_subs"] == 1].values

    # Create a figure for the boxplots
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed
    fig.suptitle(
        f"Boxplots of Deviations by Group for Feature: {feature}",
        fontweight="bold",
    )

    # Group data and labels
    all_data = [
        control_distance,
        inter_test_distance,
        exter_test_distance,
        high_test_distance,
    ]
    all_labels = [
        "Control",
        "Predominantly Internalising",
        "Predominantly Externalising",
        "Highly Dysregulated",
    ]

    # Create boxplot
    box = ax.boxplot(all_data, labels=all_labels, patch_artist=True)

    # Define colors for the boxplots
    box_colors = ["blue", "orange", "green", "red"]

    # Set colors for the boxplots
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)

    # Set y-axis label
    ax.set_ylabel("Mahalanobis Distance")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_density_distributions(feature: str, output_data: pd.DataFrame):
    # Extract mahalanobis distances for each group
    control_distance = output_data["mahalanobis_distance"][
        output_data["low_symp_test_subs"] == 1
    ].values
    inter_test_distance = output_data["mahalanobis_distance"][
        output_data["inter_test_subs"] == 1
    ].values
    exter_test_distance = output_data["mahalanobis_distance"][
        output_data["exter_test_subs"] == 1
    ].values
    high_test_distance = output_data["mahalanobis_distance"][
        output_data["high_test_subs"] == 1
    ].values

    # Create a figure with 1x3 subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    fig.suptitle(
        f"Distribution of Mahalanobis Distances by Group for Feature: {feature}"
    )

    # Define groups and their labels for easier iteration
    groups = [inter_test_distance, exter_test_distance, high_test_distance]
    labels = ["Internalising", "Externalising", "High Symptom"]

    # Plot density distributions for each group compared to control
    for ax, group_distance, label in zip(axes, groups, labels):
        sns.kdeplot(control_distance, ax=ax, color="blue", label="Control")
        sns.kdeplot(group_distance, ax=ax, color="orange", label=label)
        ax.set_title(f"{label} vs Control")
        ax.legend()

        # Calculate and plot mean and median lines
        control_mean = np.mean(control_distance)
        control_median = np.median(control_distance)
        group_mean = np.mean(group_distance)
        group_median = np.median(group_distance)

        # Add vertical lines for the mean of each distribution
        ax.axvline(
            control_mean,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Control Mean: {control_mean:.2f}",
        )
        ax.axvline(
            group_mean,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"{label} Mean: {group_mean:.2f}",
        )

        # Add vertical lines for the median of each distribution
        ax.axvline(
            control_median,
            color="blue",
            linestyle=":",
            linewidth=2,
            label=f"Control Median: {control_median:.2f}",
        )
        ax.axvline(
            group_median,
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"{label} Median: {group_median:.2f}",
        )

    # Set x-axis label
    for ax in axes:
        ax.set_xlabel("Mahalanobis Distance")

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def convert_p_to_significance_FDR_corrected(df):
    """Apply FDR correction to p-values from a DataFrame and categorize them into
    significance levels.


    Args:
        df (DataFrame): A pandas DataFrame containing the columns 'latent_dim' and
        'p_value', where 'latent_dim' represents the index of latent dimensions and
        'p_value' contains the corresponding p-values.

    Returns:
        dict: A dictionary mapping 'latent_dim_{index}' to its significance level.
    """
    # Extract p-values and latent dimensions
    p_values = df["p_value"].tolist()
    latent_dims = df["latent_dim"].tolist()

    # Apply the FDR correction using Benjamini-Hochberg
    alpha = 0.05  # Base significance level
    rejected, corrected_pvals, _, _ = multipletests(
        p_values, alpha=alpha, method="fdr_bh"
    )

    # Initialize the dictionary to store significance levels
    significance_levels = {}
    for latent_dim, reject, p_value in zip(latent_dims, rejected, corrected_pvals):
        key = f"latent_dim_{latent_dim}"  # Format the key as specified
        if not reject:
            significance_levels[key] = "ns"  # Not significant
        elif p_value <= 0.001:
            significance_levels[key] = "***"  # Very highly significant
        elif p_value <= 0.01:
            significance_levels[key] = "**"  # Highly significant
        else:
            significance_levels[key] = "*"  # Significant

    return significance_levels


def plot_ind_dim_violin(feature, output_data, latent_dim, ind_dim_dev_U_test_results):
    cohort_map = {
        "inter_test_subs": "Predominantly Internalising",
        "exter_test_subs": "Predominantly Externalising",
        "high_test_subs": "Highly Dysregulated",
    }

    # Set default properties for bold text
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    for cohort_key in cohort_map:
        # Extract p-values for the current cohort
        ind_dim_p_value_df = ind_dim_dev_U_test_results[
            ind_dim_dev_U_test_results["clinical_cohort"] == cohort_key
        ]

        # Apply FDR correction and categorize p-values into significance levels
        significance_levels = convert_p_to_significance_FDR_corrected(
            ind_dim_p_value_df
        )

        # Create DataFrame for plotting
        data_for_plotting = pd.DataFrame()

        for i in range(latent_dim):
            normative_deviation = output_data[f"latent_deviation_{i}"][
                output_data["low_symp_test_subs"] == 1
            ]
            clinical_deviation = output_data[f"latent_deviation_{i}"][
                output_data[cohort_key] == 1
            ]

            norm_df = pd.DataFrame(
                {
                    "Value": normative_deviation,
                    "Type": "Control",
                    "Latent Dimension": f"Dimension {i+1}",
                }
            )
            clin_df = pd.DataFrame(
                {
                    "Value": clinical_deviation,
                    "Type": cohort_map[cohort_key],
                    "Latent Dimension": f"Dimension {i+1}",
                }
            )
            combined_df = pd.concat([norm_df, clin_df])
            data_for_plotting = pd.concat(
                [data_for_plotting, combined_df], ignore_index=True
            )

        # Plot the data using seaborn's violinplot
        plt.figure(figsize=(20, 10))
        ax = sns.violinplot(
            x="Latent Dimension",
            y="Value",
            hue="Type",
            data=data_for_plotting,
            split=True,
            inner="quartile",
            palette="muted",
        )

        # Customize the plot with dynamic titles and larger x-axis labels
        title = f"Deviation at Individual Latent Dimensions of {feature} - {cohort_map[cohort_key]} vs Control"
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Latent Dimensions", fontsize=14)
        ax.set_ylabel("Deviation", fontsize=14)
        ax.tick_params(axis="x", labelsize=14)  # Adjust x-axis label size

        # Annotate significance on top of each violin pair
        for i in range(latent_dim):
            significance = significance_levels[f"latent_dim_{i}"]
            y_max = data_for_plotting[
                data_for_plotting["Latent Dimension"] == f"Dimension {i+1}"
            ].Value.max()
            ax.text(
                i,
                y_max * 1.25,
                significance,
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=12,
                color="black",
                weight="bold",
            )

        plt.legend(title="Group", title_fontsize="13", fontsize="12")
        plt.tight_layout()
        plt.show()


def plot_correlations(
    feature,
    output_data,
    metric="mahalanobis_distance",
):
    # Extract data for plotting
    x = output_data["cbcl_scr_syn_totprob_t"]
    y = output_data[metric]

    # Compute the linear regression and correlation
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Create scatter plot

    metric_names = {
        "mahalanobis_distance": "Mahalanobis Distance",
        "latent_deviation": "Latent Deviation",
        "reconstruction_deviation": "Reconstruction Deviation",
        "standardised_reconstruction_deviation": "Reconstruction Deviation Z-Score",
    }

    plt.figure(figsize=(10, 6))
    plt.scatter(
        x,
        y,
        alpha=0.6,
        label="Deviations vs. CBCL Total Problem T-Score",
    )

    # Add regression line
    plt.plot(
        x,
        intercept + slope * x,
        "r",
        label=f"Fit line: r={r_value:.2f} p={p_value:.4f}",
    )

    # Labeling the axes
    plt.xlabel("CBCL Score Total Problem T-Score")
    plt.ylabel(metric)
    plt.title(f"Scatter Plot of Deviations against CBCL Score for {feature}")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()
