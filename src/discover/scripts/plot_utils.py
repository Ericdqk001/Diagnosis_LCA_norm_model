import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_histograms(
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

    # Plot distributions for all groups
    axes[0, 0].hist(
        control_distance,
        bins=20,
        alpha=0.7,
        color="blue",
        label="Control",
    )
    axes[0, 0].hist(
        inter_test_distance,
        bins=20,
        alpha=0.7,
        color="orange",
        label="Internalising",
    )
    axes[0, 0].hist(
        exter_test_distance,
        bins=20,
        alpha=0.7,
        color="green",
        label="Externalising",
    )
    axes[0, 0].hist(
        high_test_distance,
        bins=20,
        alpha=0.7,
        color="red",
        label="High Symptom",
    )
    axes[0, 0].set_title("All Groups")
    axes[0, 0].legend()

    # Define colors for the vertical lines
    line_colors = ["blue", "orange", "green", "red"]

    # Add vertical lines for the mean of each distribution in the first subplot
    for group_distance, color in zip(
        [
            control_distance,
            inter_test_distance,
            exter_test_distance,
            high_test_distance,
        ],
        line_colors,
    ):
        axes[0, 0].axvline(
            np.mean(group_distance),
            color=color,
            linestyle="--",
            label="{} Mean".format(color.capitalize()),
        )

    # Plot distributions for each group compared to control
    for ax, group_distance, group_name in zip(
        axes.flat[1:],
        [inter_test_distance, exter_test_distance, high_test_distance],
        ["Internalising", "Externalising", "High Symptom"],
    ):
        ax.hist(
            control_distance,
            bins=20,
            alpha=0.7,
            color="blue",
            label="Control",
        )
        ax.hist(
            group_distance,
            bins=20,
            alpha=0.7,
            color="red",
            label=group_name,
        )
        ax.set_title("{} vs Control".format(group_name))
        ax.legend()

        # Add vertical lines for the mean of each distribution
        ax.axvline(
            np.mean(control_distance),
            color="blue",
            linestyle="--",
            # label="Control Mean",
        )
        ax.axvline(
            np.mean(group_distance),
            color="red",
            linestyle="--",
            # label="{} Mean".format(group_name),
        )

    # Set x-axis label
    for ax in axes.flat:
        ax.set_xlabel("Mahalanobis Distance")
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


def plot_density_distributions(
    feature: str,
    control_distance: np.ndarray,
    internalising_distance: np.ndarray,
    exter_test_distance: np.ndarray,
    high_test_distance: np.ndarray,
):
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle(
        "Distribution of Mahalanobis Distances by Group for Feature: {}".format(feature)
    )

    # Plot estimated density distributions for all groups
    sns.kdeplot(control_distance, ax=axes[0, 0], color="blue", label="Control")
    sns.kdeplot(
        internalising_distance, ax=axes[0, 0], color="orange", label="Internalising"
    )
    sns.kdeplot(
        exter_test_distance, ax=axes[0, 0], color="green", label="Externalising"
    )
    sns.kdeplot(high_test_distance, ax=axes[0, 0], color="red", label="High Symptom")
    axes[0, 0].set_title("All Groups")
    axes[0, 0].legend()

    # Add vertical lines for the mean of each distribution in the first subplot
    for group_distance, colour in zip(
        [
            control_distance,
            internalising_distance,
            exter_test_distance,
            high_test_distance,
        ],
        ["blue", "orange", "green", "red"],
    ):
        group_mean = np.mean(group_distance)
        axes[0, 0].axvline(
            group_mean,
            color=colour,
            linestyle="--",
            label="{} Mean: {:.2f}".format(colour.capitalize(), group_mean),
        )

    # Plot estimated density distributions for each group compared to control
    for ax, group_distance, group_name in zip(
        axes.flat[1:],
        [internalising_distance, exter_test_distance, high_test_distance],
        ["Internalising", "Externalising", "High Symptom"],
    ):
        sns.kdeplot(control_distance, ax=ax, color="blue", label="Control")
        sns.kdeplot(group_distance, ax=ax, color="orange", label=group_name)
        ax.set_title("{} vs Control".format(group_name))
        ax.legend()

        # Add vertical lines for the mean of each distribution
        control_mean = np.mean(control_distance)
        group_mean = np.mean(group_distance)
        ax.axvline(
            control_mean,
            color="blue",
            linestyle="--",
            label="Control Mean: {:.2f}".format(control_mean),
        )
        ax.axvline(
            group_mean,
            color="orange",
            linestyle="--",
            label="{} Mean: {:.2f}".format(group_name, group_mean),
        )

    # Set x-axis label
    for ax in axes.flat:
        ax.set_xlabel("Mahalanobis Distance")

    # Adjust layout
    plt.tight_layout()
    plt.show()
