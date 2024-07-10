from pathlib import Path

import pandas as pd
from nilearn import datasets

# Fetch surface atlas
fsaverage = datasets.fetch_surf_fsaverage()

# Fetch Destrieux atlas
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
parcellation_left = destrieux_atlas["map_left"]
parcellation_right = destrieux_atlas["map_right"]
labels = destrieux_atlas["labels"]

# Load the relevant ROI labels

modalities = [
    "cortical_thickness",
    "cortical_volume",
    "cortical_surface_area",
    "rsfmri",
]

clinical_cohorts = [
    "inter_test",
    "exter_test",
    "high_test",
]

for modality in modalities:

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
