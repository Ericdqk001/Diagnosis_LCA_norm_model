from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

# Path to the CSV file
ind_dim_dev_U_test_results_path = Path(
    "src",
    "discover",
    "results",
    "all_features_all_ind_dim_dev_U_test_results.csv",
)

# Load the DataFrame
ind_dim_dev_U_test_results_df = pd.read_csv(ind_dim_dev_U_test_results_path)

# Define feature sets and clinical cohorts
feature_sets = [
    "cortical_thickness",
    "cortical_volume",
    "cortical_surface_area",
    "rsfmri",
]

clinical_cohorts = [
    "inter_test_subs",
    "exter_test_subs",
    "high_test_subs",
]

# Iterate over each feature and clinical cohort
for feature in feature_sets:
    for clinical_cohort in clinical_cohorts:

        # Filter DataFrame to get relevant p-values
        filter_mask = (ind_dim_dev_U_test_results_df["Feature"] == feature) & (
            ind_dim_dev_U_test_results_df["clinical_cohort"] == clinical_cohort
        )
        cohort_feature_p_values = ind_dim_dev_U_test_results_df[filter_mask]["p_value"]

        # Perform FDR correction
        rejected, pvals_corrected, _, _ = multipletests(
            cohort_feature_p_values, alpha=0.05, method="fdr_bh"
        )

        # Get indices of significant tests
        significant_indices = ind_dim_dev_U_test_results_df[filter_mask][rejected].index

        # Retrieve latent dimensions using significant indices
        significant_latent_dims = ind_dim_dev_U_test_results_df.loc[
            significant_indices, "latent_dim"
        ]

        if len(significant_latent_dims.tolist()) != 0:

            print(
                f"Feature: {feature}, Cohort: {clinical_cohort}, Significant Latent Dimensions: {significant_latent_dims.tolist()}"
            )


# p-value 0.05046615 for dimension 10 of cortical thickness for high symptom subjects
