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

cortical_feature_pass_path = Path(
    "data",
    "processed_data",
    "t1w_cortical_thickness_bl_pass.csv",
)

neuroimaging_sample_subs = pd.read_csv(
    cortical_feature_pass_path,
    index_col=0,
    low_memory=False,
).index.tolist()


filtered_lca_class_memberships = lca_class_memberships[
    lca_class_memberships.index.isin(neuroimaging_sample_subs)
]

lca_psych_dx = filtered_lca_class_memberships.join(
    psych_dx,
    how="inner",
)

# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["axes.titleweight"] = "bold"


from pathlib import Path

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

cortical_feature_pass_path = Path(
    "data",
    "processed_data",
    "t1w_cortical_thickness_bl_pass.csv",
)

neuroimaging_sample_subs = pd.read_csv(
    cortical_feature_pass_path,
    index_col=0,
    low_memory=False,
).index.tolist()


filtered_lca_class_memberships = lca_class_memberships[
    lca_class_memberships.index.isin(neuroimaging_sample_subs)
]

lca_psych_dx = filtered_lca_class_memberships.join(
    psych_dx,
    how="inner",
)

# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["axes.titleweight"] = "bold"


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

    print(len(df))

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

    class_ids_to_plot = [2, 3, 4]
    colors = ["green", "red", "purple"]

    # Loop through the classes and plot separately
    for i, class_id in enumerate(class_ids_to_plot):
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6), dpi=100)

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

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_diagnoses, rotation=45, fontsize=12)

        # Set radial limits explicitly to ensure consistency
        ax.set_ylim([0, 1])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="black", size=10)

        ax.grid(True)

        # Save the plot to the specified path with no title
        plot_filename = f"src/LCA/images/radar_chart_class_{class_id}_DX.png"
        plt.savefig(plot_filename, bbox_inches="tight", dpi=300)

        plt.close()  # Close the figure to avoid overlapping plots

    print("Radar charts saved successfully.")


# Example usage
plot_diagnosis_proportion_radar(
    lca_psych_dx,
    low_entropy=True,
)
