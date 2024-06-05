from pathlib import Path

import pandas as pd
import torch
from discover.scripts.plot_utils import plot_ind_dim_violin
from discover.scripts.test_utils import (
    get_individual_deviation_p_values,
)

# welch_t_test_p_values,

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cVAE_discover_results_path = Path(
    "src",
    "discover",
    "results",
)


feature_sets = {
    "t1w_cortical_thickness_rois": "Cortical Thickness",
    "t1w_cortical_volume_rois": "Cortical Volume",
    "t1w_cortical_surface_area_rois": "Cortical Surface Area",
}

latent_dim = 10


def discover(if_low_entropy: bool = False):

    output_data_save_path = Path(
        cVAE_discover_results_path,
        "output_data",
    )

    if if_low_entropy:
        output_data_save_path = Path(
            cVAE_discover_results_path,
            "output_data_low_entropy",
        )

    all_ind_dim_dev_U_test_results = []

    ind_dim_dev_test_norm_assump_results = []

    ind_dim_test_var_assump_results = []

    for feature in feature_sets:

        print(f"Discovering feature: {feature}")

        feature_output_data_save_path = Path(
            output_data_save_path,
            f"{feature}_output_data_with_dev.csv",
        )

        feature_output_data = pd.read_csv(
            feature_output_data_save_path,
            index_col=0,
        )

        # TODO TEST it (Done)
        ind_dim_dev_U_test_results, normality_df, variance_df = (
            get_individual_deviation_p_values(
                feature_sets[feature],
                feature_output_data,
                latent_dim,
            )
        )

        print(ind_dim_dev_U_test_results)

        # Plot the violin plots

        plot_ind_dim_violin(
            feature_sets[feature],
            feature_output_data,
            latent_dim,
            ind_dim_dev_U_test_results,
        )

        # Add the DataFrame to the list

        all_ind_dim_dev_U_test_results.append(ind_dim_dev_U_test_results)

        ind_dim_dev_test_norm_assump_results.append(normality_df)

        ind_dim_test_var_assump_results.append(variance_df)

    # Combine all DataFrames into one
    all_ind_dim_dev_U_test_results_df = pd.concat(all_ind_dim_dev_U_test_results)

    ind_dim_dev_test_norm_assump_results_df = pd.concat(
        ind_dim_dev_test_norm_assump_results
    )

    ind_dim_test_var_assump_results_df = pd.concat(ind_dim_test_var_assump_results)

    ind_dim_test_results_path = Path(
        cVAE_discover_results_path,
        "ind_dim_test_results",
    )

    if if_low_entropy:
        ind_dim_test_results_path = Path(
            cVAE_discover_results_path,
            "low_entropy_ind_dim_test_results",
        )

    if not ind_dim_test_results_path.exists():
        ind_dim_test_results_path.mkdir(parents=True)

    # Save the combined DataFrame to a CSV file
    all_ind_dim_dev_U_test_results_df.to_csv(
        Path(
            ind_dim_test_results_path,
            "all_features_all_ind_dim_dev_U_test_results.csv",
        )
    )

    ind_dim_dev_test_norm_assump_results_df.to_csv(
        Path(
            ind_dim_test_results_path,
            "all_features_ind_dim_dev_test_norm_assump_results.csv",
        )
    )

    ind_dim_test_var_assump_results_df.to_csv(
        Path(
            ind_dim_test_results_path,
            "all_features_ind_dim_test_var_assump_results.csv",
        )
    )


if __name__ == "__main__":

    discover(if_low_entropy=False)

    # TODO Test the normality and equal variance assumptions test pipelines
