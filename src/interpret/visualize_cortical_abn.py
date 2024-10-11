from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, plotting

# Fetch surface atlas
fsaverage = datasets.fetch_surf_fsaverage()

# Fetch Destrieux atlas
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
parcellation_left = destrieux_atlas["map_left"]
parcellation_right = destrieux_atlas["map_right"]
labels = destrieux_atlas["labels"]

# Load the relevant ROI labels
cortex_modalities = [
    "cortical_thickness",
    "cortical_volume",
    "cortical_surface_area",
]

clinical_cohorts = [
    "inter_test",
    "exter_test",
    "high_test",
]

low_entropy = True

# Set the global min and max effect size
global_min_effect_size = 0.0
global_max_effect_size = 0.15

# Create a darker version of the "Reds" colormap by truncating it
cmap = plt.get_cmap("Reds")
dark_red_cmap = LinearSegmentedColormap.from_list(
    "dark_red", cmap(np.linspace(0.5, 1, 256))
)

for modality in cortex_modalities:
    print(f"Visualising {modality} results")

    if low_entropy:
        feature_brain_region_results_path = Path(
            "src",
            "discover",
            "results",
            "low_entropy",
            "sig_ind_regions",
            f"{modality}_significant_regions.csv",
        )
    else:
        feature_brain_region_results_path = Path(
            "src",
            "discover",
            "results",
            "sig_ind_regions",
            f"{modality}_significant_regions.csv",
        )

    feature_brain_region_results_df = pd.read_csv(feature_brain_region_results_path)

    # Adjust the effect size based on the threshold conditions
    feature_brain_region_results_df["Mean Effect Size"] = (
        feature_brain_region_results_df["Mean Effect Size"].apply(
            lambda x: 0 if x < 0.15 else 0.15
        )
    )

    # Normalize the effect size according to global min and max
    feature_brain_region_results_df["Mean Effect Size"] = (
        feature_brain_region_results_df["Mean Effect Size"] - global_min_effect_size
    ) / (global_max_effect_size - global_min_effect_size)

    for cohort in clinical_cohorts:
        print(f"Visualising {cohort} cohort")

        positive_brain_region_df = feature_brain_region_results_df[
            feature_brain_region_results_df["Group"] == cohort
        ]

        roi_labels = positive_brain_region_df["Brain Region"].tolist()
        roi_effect_sizes = positive_brain_region_df["Mean Effect Size"].tolist()

        # Splitting into left and right ROI labels
        left_roi_labels = [name[2:] for name in roi_labels if name.startswith("L ")]
        right_roi_labels = [name[2:] for name in roi_labels if name.startswith("R ")]

        left_effect_sizes = [
            roi_effect_sizes[idx]
            for idx, name in enumerate(roi_labels)
            if name.startswith("L ")
        ]
        right_effect_sizes = [
            roi_effect_sizes[idx]
            for idx, name in enumerate(roi_labels)
            if name.startswith("R ")
        ]

        pcc_left_labels = [
            labels.index(label.encode("utf-8")) for label in left_roi_labels
        ]

        pcc_right_labels = [
            labels.index(label.encode("utf-8")) for label in right_roi_labels
        ]

        # Create masks for multiple regions of interest in the right hemisphere and left hemisphere
        pcc_left_mask = np.zeros_like(parcellation_left, dtype=float)
        pcc_right_mask = np.zeros_like(parcellation_right, dtype=float)

        for idx, label in enumerate(pcc_left_labels):
            pcc_left_mask[parcellation_left == label] = left_effect_sizes[idx]

        for idx, label in enumerate(pcc_right_labels):
            pcc_right_mask[parcellation_right == label] = right_effect_sizes[idx]

        # Load the fsaverage5 pial surface for left and right hemispheres
        fsaverage_pial_left = fsaverage["pial_left"]
        fsaverage_pial_right = fsaverage["pial_right"]

        # Create a figure with 1x4 subplots
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={"projection": "3d"})

        # Plotting the left hemisphere, lateral view
        plot1 = plotting.plot_surf_roi(
            fsaverage_pial_left,
            roi_map=pcc_left_mask,
            hemi="left",
            view="lateral",
            bg_map=fsaverage["sulc_left"],
            bg_on_data=True,
            colorbar=False,
            cmap=dark_red_cmap,
            axes=axes[0],
        )

        # Plotting the left hemisphere, medial view
        plot2 = plotting.plot_surf_roi(
            fsaverage_pial_left,
            roi_map=pcc_left_mask,
            hemi="left",
            view="medial",
            bg_map=fsaverage["sulc_left"],
            bg_on_data=True,
            colorbar=False,
            cmap=dark_red_cmap,
            axes=axes[1],
        )

        # Plotting the right hemisphere, lateral view
        plot3 = plotting.plot_surf_roi(
            fsaverage_pial_right,
            roi_map=pcc_right_mask,
            hemi="right",
            view="lateral",
            bg_map=fsaverage["sulc_right"],
            bg_on_data=True,
            colorbar=False,
            cmap=dark_red_cmap,
            axes=axes[2],
        )

        # Plotting the right hemisphere, medial view
        plot4 = plotting.plot_surf_roi(
            fsaverage_pial_right,
            roi_map=pcc_right_mask,
            hemi="right",
            view="medial",
            bg_map=fsaverage["sulc_right"],
            bg_on_data=True,
            colorbar=False,
            cmap=dark_red_cmap,
            axes=axes[3],
        )

        # Adjust layout
        plt.tight_layout()

        if low_entropy:
            image_save_path = Path(
                "src",
                "interpret",
                "plots",
                "low_entropy",
            )
        else:
            image_save_path = Path(
                "src",
                "interpret",
                "plots",
                "original",
            )

        if not image_save_path.exists():
            image_save_path.mkdir(parents=True)

        if low_entropy:
            fig.savefig(Path(image_save_path, f"{cohort}_{modality}.png"))
        else:
            fig.savefig(Path(image_save_path, f"{cohort}_{modality}_original.png"))

        # Show the plot
        plt.show()
