import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set global parameters to make all text bold
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


# def plot_histograms(
#     feature: str,
#     output_data: pd.DataFrame,
# ):

#     control_distance = output_data["mahalanobis_distance"][
#         output_data["low_symp_test_subs"] == 1
#     ].values

#     inter_test_distance = output_data["mahalanobis_distance"][
#         output_data["inter_test_subs"] == 1
#     ].values

#     exter_test_distance = output_data["mahalanobis_distance"][
#         output_data["exter_test_subs"] == 1
#     ].values

#     high_test_distance = output_data["mahalanobis_distance"][
#         output_data["high_test_subs"] == 1
#     ].values

#     # Create a figure with subplots
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
#     fig.suptitle(
#         "Distribution of Mahalanobis Distances by Group for Feature: {}".format(feature)
#     )

#     # Plot distributions for all groups
#     axes[0, 0].hist(
#         control_distance,
#         bins=20,
#         alpha=0.7,
#         color="blue",
#         label="Control",
#     )
#     axes[0, 0].hist(
#         inter_test_distance,
#         bins=20,
#         alpha=0.7,
#         color="orange",
#         label="Internalising",
#     )
#     axes[0, 0].hist(
#         exter_test_distance,
#         bins=20,
#         alpha=0.7,
#         color="green",
#         label="Externalising",
#     )
#     axes[0, 0].hist(
#         high_test_distance,
#         bins=20,
#         alpha=0.7,
#         color="red",
#         label="High Symptom",
#     )
#     axes[0, 0].set_title("All Groups")
#     axes[0, 0].legend()

#     # Define colors for the vertical lines
#     line_colors = ["blue", "orange", "green", "red"]

#     # Add vertical lines for the mean of each distribution in the first subplot
#     for group_distance, color in zip(
#         [
#             control_distance,
#             inter_test_distance,
#             exter_test_distance,
#             high_test_distance,
#         ],
#         line_colors,
#     ):
#         axes[0, 0].axvline(
#             np.mean(group_distance),
#             color=color,
#             linestyle="--",
#             label="{} Mean".format(color.capitalize()),
#         )

#     # Plot distributions for each group compared to control
#     for ax, group_distance, group_name in zip(
#         axes.flat[1:],
#         [inter_test_distance, exter_test_distance, high_test_distance],
#         ["Internalising", "Externalising", "High Symptom"],
#     ):
#         ax.hist(
#             control_distance,
#             bins=20,
#             alpha=0.7,
#             color="blue",
#             label="Control",
#         )
#         ax.hist(
#             group_distance,
#             bins=20,
#             alpha=0.7,
#             color="red",
#             label=group_name,
#         )
#         ax.set_title("{} vs Control".format(group_name))
#         ax.legend()

#         # Add vertical lines for the mean of each distribution
#         ax.axvline(
#             np.mean(control_distance),
#             color="blue",
#             linestyle="--",
#             # label="Control Mean",
#         )
#         ax.axvline(
#             np.mean(group_distance),
#             color="red",
#             linestyle="--",
#             # label="{} Mean".format(group_name),
#         )

#     # Set x-axis label
#     for ax in axes.flat:
#         ax.set_xlabel("Mahalanobis Distance")
#     # Adjust layout
#     plt.tight_layout()
#     plt.show()


def plot_histograms(feature: str, output_data: pd.DataFrame, p_values_df: pd.DataFrame):
    # Extract distances for each group
    control_distance = output_data["mahalanobis_distance"][
        output_data["low_symp_test_subs"] == 1
    ].values
    internalising_distance = output_data["mahalanobis_distance"][
        output_data["inter_test_subs"] == 1
    ].values
    exter_test_distance = output_data["mahalanobis_distance"][
        output_data["exter_test_subs"] == 1
    ].values
    high_test_distance = output_data["mahalanobis_distance"][
        output_data["high_test_subs"] == 1
    ].values

    # Create a figure with subplots (1 row, 3 columns)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    fig.suptitle(
        "Distribution of Mahalanobis Distances by Group for Feature: {}".format(
            feature
        ),
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
            label=group_name,
            histtype="stepfilled",
        )
        ax.set_title("{} vs Control".format(group_name))

        # Add vertical lines for the median of each distribution
        control_median_line = ax.axvline(
            np.median(control_distance),
            color="grey",
            linestyle="--",
            label="Control Median",
        )
        group_median_line = ax.axvline(
            np.median(group_distance),
            color=color,
            linestyle="--",
            label=f"{group_name} Median",
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

        ax.set_xlabel("Mahalanobis Distance")
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_boxplots(
    feature: str,
    output_data: pd.DataFrame,
):

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

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle(
        "Distribution of Mahalanobis Distances by Group for Feature: {}".format(feature)
    )

    # Plot boxplots for all groups
    all_data = [
        control_distance,
        inter_test_distance,
        exter_test_distance,
        high_test_distance,
    ]
    all_labels = ["Control", "Internalising", "Externalising", "High Symptom"]
    axes[0, 0].boxplot(all_data, labels=all_labels, patch_artist=True)
    axes[0, 0].set_title("All Groups")

    # Define colors for the boxplots
    box_colors = ["blue", "orange", "green", "red"]

    # Set colors for the boxplots in the first subplot
    for patch, color in zip(axes[0, 0].artists, box_colors):
        patch.set_facecolor(color)

    # Plot boxplots for each group compared to control
    for ax, group_distance, group_name, color in zip(
        axes.flat[1:],
        [inter_test_distance, exter_test_distance, high_test_distance],
        ["Internalising", "Externalising", "High Symptom"],
        ["orange", "green", "red"],
    ):
        ax.boxplot(
            [control_distance, group_distance],
            labels=["Control", group_name],
            patch_artist=True,
        )
        ax.set_title("{} vs Control".format(group_name))

        # Set colors for the boxplots in the current subplot
        for patch, col in zip(ax.artists, ["blue", color]):
            patch.set_facecolor(col)

    # Set y-axis label
    for ax in axes.flat:
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
