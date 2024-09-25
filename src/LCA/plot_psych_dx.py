from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

psych_dx_path = Path(
    "data",
    "liza_data",
    "all_psych_dx_r5.csv",
)

psych_dx = pd.read_csv(
    psych_dx_path,
    index_col=0,
    low_memory=False,
)

lca_path = Path(
    "data",
    "LCA",
)

lca_class_memberships_path = Path(
    lca_path,
    "cbcl_class_member_prob.csv",
)

lca_class_memberships = pd.read_csv(
    lca_class_memberships_path,
    index_col=0,
    low_memory=False,
)

lca_class_memberships_low_entropy = lca_class_memberships[
    lca_class_memberships["entropy"] <= 1
]

lca_psych_dx_low_entropy = lca_class_memberships_low_entropy.join(
    psych_dx,
    how="inner",
)

# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["axes.titleweight"] = "bold"


def plot_diagnosis_proportion(df, low_entropy=False):
    diagnoses = [
        "Has_ADHD",
        "Has_Depression",
        "Has_Bipolar",
        "Has_Anxiety",
        "Has_OCD",
        "Has_ASD",
        "Has_DBD",
    ]

    # Shorten diagnosis names for plotting
    short_diagnoses = [diag.replace("Has_", "") for diag in diagnoses]

    class_names = [
        "Low Symptom",
        "Predominantly Internalising",
        "Predominantly Externalising",
        "Highly Dysregulated",
    ]

    if low_entropy:
        # Remove subjects with high entropy
        high_entropy_subs_path = Path(
            "data",
            "LCA",
            "subjects_with_high_entropy.csv",
        )
        high_entropy_subs = pd.read_csv(
            high_entropy_subs_path,
            low_memory=False,
        )["subject"].tolist()
        df = df[~df.index.isin(high_entropy_subs)]

    max_proportion_class_4 = df[df["predicted_class"] == 4][diagnoses].mean().max()

    # Set up the plot in a 2x2 grid
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 14),
        sharey=True,  # Use the same y-scale for all subplots
    )

    class_ids_to_plot = [1, 2, 3, 4]  # Include all classes

    for i, class_id in enumerate(class_ids_to_plot):
        # Determine position in the 2x2 grid using integer division and modulus
        ax = axes[i // 2, i % 2]
        class_df = df[df["predicted_class"] == class_id]
        proportions = class_df[diagnoses].mean()
        errors = class_df[diagnoses].std() / np.sqrt(
            class_df.shape[0]
        )  # Standard error

        ax.bar(
            short_diagnoses,
            proportions,
            yerr=errors,
            color="#4c72b0",
            capsize=3,
            edgecolor="black",
        )
        ax.set_title(f"{class_names[class_id - 1]}", fontweight="bold", fontsize=16)
        ax.set_xticklabels(short_diagnoses, rotation=45, fontsize=14)
        ax.set_xlabel("Diagnosis", fontsize=14, fontweight="bold")
        ax.grid(False)

        if i % 2 == 0:  # Add y-label to the first column only
            ax.set_ylabel("Proportion", fontsize=14, fontweight="bold")

    # Set the limit of the y-axis to the maximum proportion observed in class 4
    plt.ylim(0, max_proportion_class_4 + 0.1)

    # Overall title for the figure
    fig.suptitle(
        "Proportion of Diagnoses Across Different Latent Classes",
        fontsize=18,
        fontweight="bold",
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )  # Adjust layout to make room for the overall title
    plt.show()


# plot_diagnosis_proportion(lca_psych_dx_low_entropy, low_entropy=True)


def plot_diagnosis_proportion_radar(df, low_entropy=False):
    diagnoses = [
        "Has_ADHD",
        "Has_Depression",
        "Has_Bipolar",
        "Has_Anxiety",
        "Has_OCD",
        "Has_ASD",
        "Has_DBD",
    ]

    short_diagnoses = [diag.replace("Has_", "") for diag in diagnoses]

    class_names = [
        "Predominantly Internalizing",
        "Predominantly Externalizing",
        "Highly Dysregulated",
    ]

    if low_entropy:
        high_entropy_subs_path = Path(
            "data",
            "LCA",
            "subjects_with_high_entropy.csv",
        )
        high_entropy_subs = pd.read_csv(
            high_entropy_subs_path,
            low_memory=False,
        )["subject"].tolist()
        df = df[~df.index.isin(high_entropy_subs)]

    num_vars = len(diagnoses)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        1, 3, subplot_kw=dict(polar=True), figsize=(18, 6), dpi=100
    )

    class_ids_to_plot = [2, 3, 4]
    colors = ["green", "red", "purple"]

    for i, class_id in enumerate(class_ids_to_plot):
        ax = axes[i]
        class_df = df[df["predicted_class"] == class_id]
        proportions = class_df[diagnoses].mean().tolist()
        proportions += proportions[:1]

        ax.plot(
            angles,
            proportions,
            "o-",
            color=colors[i],
            alpha=0.8,
            linewidth=3,
        )
        ax.fill(angles, proportions, color=colors[i], alpha=0.3)

        ax.set_title(f"{class_names[i]}", fontweight="bold", fontsize=16)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_diagnoses, rotation=45, fontsize=12)

        # Set radial limits explicitly to ensure consistency
        ax.set_ylim([0, 1])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="black", size=10)

        ax.grid(True)

    fig.suptitle("", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Example usage
plot_diagnosis_proportion_radar(
    lca_psych_dx_low_entropy,
    low_entropy=True,
)
