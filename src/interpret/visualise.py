# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import datasets, plotting

# Fetch surface atlas
fsaverage = datasets.fetch_surf_fsaverage()

# Fetch Destrieux atlas
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
parcellation_left = destrieux_atlas["map_left"]
parcellation_right = destrieux_atlas["map_right"]
labels = destrieux_atlas["labels"]


# Load the relevant ROI labels

cortical_feature = "t1w_cortical_thickness_rois"

feature_brain_region_results_path = Path(
    "src",
    "interpret",
    "results",
    f"{cortical_feature}",
)

clinical_cohorts = [
    "inter_test",
    "exter_test",
    "high_test",
]

for cohort in clinical_cohorts:
    print(f"Visualising {cohort} cohort")
    positive_brain_region_path = Path(
        feature_brain_region_results_path,
        f"{cohort}_positive_brain_regions.csv",
    )

    positive_brain_region_df = pd.read_csv(positive_brain_region_path)

    roi_labels = positive_brain_region_df["brain_regions"].tolist()

    # Splitting into left and right ROI labels
    left_roi_labels = [name[2:] for name in roi_labels if name.startswith("L ")]
    right_roi_labels = [name[2:] for name in roi_labels if name.startswith("R ")]

    pcc_left_labels = [labels.index(label.encode("utf-8")) for label in left_roi_labels]

    pcc_right_labels = [
        labels.index(label.encode("utf-8")) for label in right_roi_labels
    ]

    # Create masks for multiple regions of interest in the right hemisphere and left hemisphere
    pcc_left_mask = np.isin(parcellation_left, pcc_left_labels)

    pcc_right_mask = np.isin(parcellation_right, pcc_right_labels)

    # Load the fsaverage5 pial surface for left and right hemispheres
    fsaverage_pial_left = fsaverage["pial_left"]
    fsaverage_pial_right = fsaverage["pial_right"]

    image_save_path = Path(
        "src",
        "interpret",
        "plots",
    )

    # Create a figure with 1x4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={"projection": "3d"})

    # Plotting the left hemisphere, lateral view
    plotting.plot_surf_roi(
        fsaverage_pial_left,
        roi_map=pcc_left_mask,
        hemi="left",
        view="lateral",
        bg_map=fsaverage["sulc_left"],
        bg_on_data=True,
        title="Left Hemisphere Lateral",
        colorbar=False,
        cmap="gist_rainbow",
        axes=axes[0],
    )

    # Plotting the left hemisphere, medial view
    plotting.plot_surf_roi(
        fsaverage_pial_left,
        roi_map=pcc_left_mask,
        hemi="left",
        view="medial",
        bg_map=fsaverage["sulc_left"],
        bg_on_data=True,
        title="Left Hemisphere Medial",
        colorbar=False,
        cmap="gist_rainbow",
        axes=axes[1],
    )

    # Plotting the right hemisphere, lateral view
    plotting.plot_surf_roi(
        fsaverage_pial_right,
        roi_map=pcc_right_mask,
        hemi="right",
        view="lateral",
        bg_map=fsaverage["sulc_right"],
        bg_on_data=True,
        title="Right Hemisphere Lateral",
        colorbar=False,
        cmap="gist_rainbow",
        axes=axes[2],
    )

    # Plotting the right hemisphere, medial view
    plotting.plot_surf_roi(
        fsaverage_pial_right,
        roi_map=pcc_right_mask,
        hemi="right",
        view="medial",
        bg_map=fsaverage["sulc_right"],
        bg_on_data=True,
        title="Right Hemisphere Medial",
        colorbar=False,
        cmap="gist_rainbow",
        axes=axes[3],
    )

    # Adjust layout
    plt.tight_layout()

    # Save the combined plot
    fig.savefig(Path(image_save_path, f"{cohort}_combined_plot.png"))

    # Show the plot
    plt.show()

# %%
# Plot the images together

# image_paths = [
#     Path("src/interpret/ct_plots/left_lateral.png"),
#     Path("src/interpret/ct_plots/left_medial.png"),
#     Path("src/interpret/ct_plots/right_lateral.png"),
#     Path("src/interpret/ct_plots/right_medial.png"),
# ]

# sorted_image_paths = sorted(image_paths, key=lambda x: x.stem.split("_")[1])

# # Create figure with subplots
# fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))

# # Loop over axes and image paths
# for ax, image_path in zip(axes, image_paths):
#     img = Image.open(image_path)
#     ax.imshow(img)
#     ax.axis("off")

# # Adjust layout
# plt.subplots_adjust(
#     left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01
# )
# plt.tight_layout(pad=0.1)

# # Adjust layout to ensure the plots are nicely spaced
# plt.tight_layout()
# plt.show()
