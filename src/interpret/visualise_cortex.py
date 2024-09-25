from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from PIL import Image

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

# Step 1: Determine the global min and max effect sizes across all modalities using cortical_thickness
global_min_effect_size = None
global_max_effect_size = None

for modality in cortex_modalities:
    feature_brain_region_results_path = Path(
        "src",
        "discover",
        "results",
        "low_entropy",
        "sig_ind_regions",
        f"{modality}_significant_regions.csv",
    )

    feature_brain_region_results_df = pd.read_csv(feature_brain_region_results_path)

    min_effect_size = feature_brain_region_results_df["Mean Effect Size"].min()
    max_effect_size = feature_brain_region_results_df["Mean Effect Size"].max()

    if global_min_effect_size is None or min_effect_size < global_min_effect_size:
        global_min_effect_size = min_effect_size
    if global_max_effect_size is None or max_effect_size > global_max_effect_size:
        global_max_effect_size = max_effect_size

print(f"Global min effect size: {global_min_effect_size}")
print(f"Global max effect size: {global_max_effect_size}")


def crop_to_brain(image_path, save_path, pad=5):
    with Image.open(image_path) as im:
        image_array = np.array(im)

        # Create a mask where each pixel is True if all channels are 255 (background)
        is_background = np.all(image_array == 255, axis=2)

        # Find the rows and columns that are not entirely background
        non_background_rows = np.any(~is_background, axis=1)
        non_background_cols = np.any(~is_background, axis=0)

        # Identify the first and last non-background pixel positions
        top = np.argmax(non_background_rows)
        bottom = len(non_background_rows) - np.argmax(non_background_rows[::-1])
        left = np.argmax(non_background_cols)
        right = len(non_background_cols) - np.argmax(non_background_cols[::-1])

        # Apply padding only to the left and right, ensuring we don't go out of bounds
        left = max(left - pad, 0)
        right = min(right + pad, image_array.shape[1])

        # Crop the image with padding on the left and right
        im_cropped = im.crop((left, top, right, bottom))
        im_cropped.save(save_path)


for modality in cortex_modalities:

    print(f"Visualising {modality} results")

    feature_brain_region_results_path = Path(
        "src",
        "discover",
        "results",
        "low_entropy",
        "sig_ind_regions",
        f"{modality}_significant_regions.csv",
    )

    feature_brain_region_results_df = pd.read_csv(feature_brain_region_results_path)

    # Normalize the effect size using the global min and max
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

        # Temporary directory for saving individual views
        temp_save_path = Path("temp_plots")
        if not temp_save_path.exists():
            temp_save_path.mkdir()

        views = ["lateral", "medial"]
        hemis = ["left", "right"]

        for i, hemi in enumerate(hemis):
            for j, view in enumerate(views):
                temp_image_path = (
                    temp_save_path / f"{cohort}_{modality}_{hemi}_{view}.png"
                )
                cropped_image_path = (
                    temp_save_path / f"{cohort}_{modality}_{hemi}_{view}_cropped.png"
                )

                fig_temp, ax_temp = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

                # Plotting the ROI on the surface
                plotting.plot_surf_roi(
                    fsaverage[f"pial_{hemi}"],
                    roi_map=pcc_left_mask if hemi == "left" else pcc_right_mask,
                    hemi=hemi,
                    view=view,
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True,
                    colorbar=False,
                    cmap="Reds",
                    axes=ax_temp,
                )
                fig_temp.savefig(temp_image_path)
                plt.close(fig_temp)

                # Crop the image to the brain region
                crop_to_brain(temp_image_path, cropped_image_path)

        # Now create the main figure and load the cropped images
        fig = plt.figure(figsize=(12, 12))

        # Create a GridSpec with reduced vertical space
        gs = gridspec.GridSpec(
            2, 2, wspace=0, hspace=-0.37
        )  # Negative hspace removes space between rows

        for i, hemi in enumerate(hemis):
            for j, view in enumerate(views):
                cropped_image_path = (
                    temp_save_path / f"{cohort}_{modality}_{hemi}_{view}_cropped.png"
                )
                img = Image.open(cropped_image_path)

                # Determine the correct position in the GridSpec
                ax = fig.add_subplot(gs[i, j])
                ax.imshow(img)
                ax.axis("off")

        # Save the final figure without the color bar
        final_image_path = Path(
            "src",
            "interpret",
            "plots",
            "bootstrap",
            f"{cohort}_{modality}_final.png",
        )

        if not final_image_path.parent.exists():
            final_image_path.parent.mkdir(parents=True)

        fig.savefig(final_image_path, bbox_inches="tight", pad_inches=0)
        plt.show()

        print(f"Saved final image to {final_image_path}")

# Create and save the color bar separately (do this only once)
fig_cbar, ax_cbar = plt.subplots(figsize=(6, 1))
sm = plt.cm.ScalarMappable(
    cmap="Reds",
    norm=plt.Normalize(vmin=global_min_effect_size, vmax=global_max_effect_size),
)
fig_cbar.colorbar(sm, orientation="horizontal", ax=ax_cbar)
colorbar_path = Path(
    "src",
    "interpret",
    "plots",
    "bootstrap",
    "global_colorbar.png",
)
fig_cbar.savefig(colorbar_path, bbox_inches="tight", pad_inches=0.1)
plt.close(fig_cbar)

print(f"Saved global color bar to {colorbar_path}")
