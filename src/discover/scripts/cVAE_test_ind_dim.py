from pathlib import Path

import pandas as pd
import torch
from discover.scripts.test_utils import (
    compute_distance_deviation_cVAE,
    get_individual_deviation_p_values,
    prepare_inputs_cVAE,
)

# welch_t_test_p_values,
from modelling.models.cVAE import cVAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


FEATURE_NAMES_MAP = {
    "t1w_cortical_thickness_rois": "Cortical Thickness",
    "t1w_cortical_volume_rois": "Cortical Volume",
    "gordon_net_subcor_limbic_no_dup": "Functional Connectivity",
}


processed_data_path = Path(
    "data",
    "processed_data",
)

LCA_data_path = Path(
    "data",
    "LCA",
)

data_path = Path(
    processed_data_path,
    "all_brain_features_resid_exc_sex.csv",
)

cbcl_lca_path = Path(
    LCA_data_path,
    "cbcl_class_member_prob.csv",
)

features_of_interest_path = Path(
    processed_data_path,
    "brain_features_of_interest.json",
)

data_splits_path = Path(
    processed_data_path,
    "data_splits_with_clinical_val.json",
)

cVAE_discover_results_path = Path(
    "src",
    "discover",
    "results",
)

output_data_save_path = Path(
    cVAE_discover_results_path,
    "out_put_data",
)

if not output_data_save_path.exists():
    output_data_save_path.mkdir(parents=True)


cVAE_feature_hyper = {
    "t1w_cortical_thickness_rois": {
        "learning_rate": 0.0005,
        "latent_dim": 10,
        "hidden_dim": [40],
    },
    "t1w_cortical_volume_rois": {
        "learning_rate": 0.001,
        "latent_dim": 10,
        "hidden_dim": [30, 30],
    },
    "t1w_cortical_surface_area_rois": {
        "learning_rate": 0.0005,
        "latent_dim": 10,
        "hidden_dim": [30, 30],
    },
    "gordon_net_subcor_limbic_no_dup": {
        "learning_rate": 0.001,
        "latent_dim": 10,
        "hidden_dim": [30, 30],
    },
}


def discover():

    feature_sets = {
        "t1w_cortical_thickness_rois": "cortical_thickness",
        "t1w_cortical_volume_rois": "cortical_volume",
        "t1w_cortical_surface_area_rois": "cortical_surface_area",
        "gordon_net_subcor_limbic_no_dup": "rsfmri",
    }

    all_ind_dim_dev_U_test_results = []

    ind_dim_dev_test_norm_assump_results = []

    ind_dim_test_var_assump_results = []

    for feature in feature_sets:

        print(f"Discovering feature: {feature}")

        feature_checkpoint_path = Path(
            "checkpoints",
            feature_sets[feature],
        )

        checkpoint_path = Path(feature_checkpoint_path, "model_weights_no_dx.pt")

        (
            train_dataset,
            test_dataset,
            train_cov,
            test_cov,
            input_dim,
            c_dim,
            output_data,
        ) = prepare_inputs_cVAE(
            feature,
            brain_features_path=data_path,
            cbcl_path=cbcl_lca_path,
            brain_features_of_interest_path=features_of_interest_path,
            data_splits_path=data_splits_path,
        )

        hyperparameters = cVAE_feature_hyper.get(feature)

        model = cVAE(
            input_dim=input_dim,
            hidden_dim=hyperparameters.get("hidden_dim"),
            latent_dim=hyperparameters.get("latent_dim"),
            c_dim=c_dim,
            learning_rate=hyperparameters.get("learning_rate"),
            non_linear=True,
        ).to(DEVICE)

        model.load_state_dict(torch.load(checkpoint_path))

        output_data_with_dev = compute_distance_deviation_cVAE(
            model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_cov=train_cov,
            test_cov=test_cov,
            latent_dim=hyperparameters.get("latent_dim"),
            output_data=output_data,
        )

        feature_output_data_save_path = Path(
            output_data_save_path,
            f"{feature}_output_data_with_dev.csv",
        )

        output_data_with_dev.to_csv(feature_output_data_save_path)

        ind_dim_dev_U_test_results, normality_df, variance_df = (
            get_individual_deviation_p_values(
                output_data_with_dev,
                hyperparameters.get("latent_dim"),
            )
        )

        # Add the DataFrame to the list

        ind_dim_dev_U_test_results["Feature"] = feature_sets[feature]

        normality_df["Feature"] = feature_sets[feature]

        variance_df["Feature"] = feature_sets[feature]

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

    discover()

    # TODO Test the normality and equal variance assumptions test pipelines
