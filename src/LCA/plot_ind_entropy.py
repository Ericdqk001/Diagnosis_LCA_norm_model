from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_entropy_distribution_by_class(df):

    class_names = [
        "Low Symptom",
        "Predominantly Internalising",
        "Predominantly Externalising",
        "Highly Dysregulated",
    ]

    # Set the figure size and create subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    # Iterate through each predicted class and plot the histogram
    for i in range(1, 5):
        ax = axes[(i - 1) // 2, (i - 1) % 2]  # Determine the subplot location
        class_entropy = df[df["predicted_class"] == i]["entropy"]

        # Calculate the percentage of individuals with entropy > 0.2
        high_entropy_percentage = (
            class_entropy > 0.2
        ).mean() * 100  # mean() gives proportion, *100 to convert to percentage

        sns.histplot(class_entropy, bins=30, kde=True, ax=ax)
        ax.set_title(f"{class_names[i-1]} Class")
        ax.set_xlabel("Shannon's Entropy")
        ax.set_ylabel("Frequency")
        ax.grid(True)

        # Annotate the percentage of individuals with entropy > 0.2
        ax.text(
            0.95,
            0.95,
            f"{high_entropy_percentage:.2f}% > 0.2 (entropy)",
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

    fig.suptitle(
        "Entropy Distributions for LCA Classes", fontsize=16, fontweight="light"
    )

    # Improve layout and avoid label cut-off
    plt.tight_layout()
    plt.show()


# Load the data
LCA_data_path = Path("data", "LCA")
lca_class_prob_path = Path(LCA_data_path, "cbcl_class_member_prob.csv")
lca_class_prob = pd.read_csv(lca_class_prob_path, index_col=0, low_memory=False)

# Example usage of the function
plot_entropy_distribution_by_class(lca_class_prob)
