from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sig_ind_regions_path = Path(
    "src",
    "discover",
    "results",
    "low_entropy",
    "sig_ind_regions",
)

modalities = [
    "cortical_thickness",
    "cortical_volume",
    "cortical_surface_area",
    "rsfmri",
]

for modality in modalities:
    sig_ind_regions_modality_path = Path(
        sig_ind_regions_path,
        f"{modality}_significant_regions.csv",
    )

    print(modality)

    if modality == "rsfmri":

        modality = "Functional Connectivity"
        print(modality)

    sig_ind_regions_modality = pd.read_csv(sig_ind_regions_modality_path)

    inter_test = sig_ind_regions_modality[
        sig_ind_regions_modality["Group"] == "inter_test"
    ]

    exter_test = sig_ind_regions_modality[
        sig_ind_regions_modality["Group"] == "exter_test"
    ]

    high_test = sig_ind_regions_modality[
        sig_ind_regions_modality["Group"] == "high_test"
    ]

    # Sort each group by "Mean Effect Size"
    inter_test = inter_test.sort_values(by="Mean Effect Size", ascending=False)
    exter_test = exter_test.sort_values(by="Mean Effect Size", ascending=False)
    high_test = high_test.sort_values(by="Mean Effect Size", ascending=False)

    # Create a figure with subplots (1 row, 3 columns)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    fig.suptitle(f"Error Bars for {modality.replace('_', ' ').title()}")

    groups = {
        "inter_test": inter_test,
        "exter_test": exter_test,
        "high_test": high_test,
    }
    group_titles = [
        "Predominantly Internalising",
        "Predominantly Externalising",
        "Highly Dysregulated",
    ]

    # Plot error bars for each group
    for ax, (group_name, group_df), group_title in zip(
        axes, groups.items(), group_titles
    ):
        ci_lower = group_df["CI Lower"].values
        ci_upper = group_df["CI Upper"].values
        mean_effect_size = group_df["Mean Effect Size"].values
        brain_regions = group_df["Brain Region"].values

        # Calculate the error for the error bars
        error = [mean_effect_size - ci_lower, ci_upper - mean_effect_size]

        ax.errorbar(
            mean_effect_size,
            range(len(mean_effect_size)),
            xerr=error,
            fmt="s",  # 's' for square markers
            color="black",
            ecolor="black",
            elinewidth=1,
            capsize=3,
            label="Mean Effect Size",
        )

        ax.set_yticks(range(len(brain_regions)))
        ax.set_yticklabels(brain_regions)
        ax.invert_yaxis()  # Highest value at the top
        ax.set_title(group_title)
        ax.set_xlabel("Mean Effect Size")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
