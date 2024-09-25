from pathlib import Path

import pandas as pd

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

low_entropy = False

for modality in modalities:

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

    feature_brain_region_results_df = pd.read_csv(
        feature_brain_region_results_path,
    )

    # print(feature_brain_region_results_df.head())

    for cohort in clinical_cohorts:

        # if cohort == "high_test":

        print(f"Visualising {modality} results for {cohort}")

        cohort_feature_brain_region_results_df = feature_brain_region_results_df[
            feature_brain_region_results_df["Group"] == cohort
        ]

        # Print regions with an effect size larger than or equal to 0.15

        print(
            cohort_feature_brain_region_results_df[
                cohort_feature_brain_region_results_df["Mean Effect Size"] >= 0.15
            ]["Variable Name"].unique()
        )

        print(
            cohort_feature_brain_region_results_df[
                cohort_feature_brain_region_results_df["Mean Effect Size"] >= 0.15
            ]["Mean Effect Size"].unique()
        )
