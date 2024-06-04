import json
from pathlib import Path

import torch

# from discover.scripts.plot_utils import (
#     plot_boxplots,
#     plot_histograms,
# )
from discover.scripts.test_utils import (
    compute_interpret_distance_deviation,
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


cVAE_feature_hyper = {
    # "t1w_cortical_thickness_rois": {
    #     "learning_rate": 0.0005,
    #     "latent_dim": 10,
    #     "hidden_dim": [40],
    # },
    # "t1w_cortical_volume_rois": {
    #     "learning_rate": 0.001,
    #     "latent_dim": 10,
    #     "hidden_dim": [30, 30],
    # },
    "t1w_cortical_surface_area_rois": {
        "learning_rate": 0.0005,
        "latent_dim": 1,
        "hidden_dim": [10, 10],
    },
    # "gordon_net_subcor_limbic_no_dup": {
    #     "learning_rate": 0.001,
    #     "latent_dim": 10,
    #     "hidden_dim": [30, 30],
    # },
}

interpret_results_path = Path(
    "src",
    "interpret",
    "results",
    "dim_4_surface_area_indices_by_sex.json",
)

with open(interpret_results_path, "r") as f:
    interpret_results = json.load(f)

relevant_feature_indices = interpret_results["Unique_indices"]

checkpoint_name = "interpret_dim_4_sa_latent_dim_1.pt"


def discover():

    feature_sets = {
        # "t1w_cortical_thickness_rois": "cortical_thickness",
        # "t1w_cortical_volume_rois": "cortical_volume",
        "t1w_cortical_surface_area_rois": "cortical_surface_area",
        # "gordon_net_subcor_limbic_no_dup": "rsfmri",
    }

    all_ind_dim_dev_U_test_results = []

    for feature in feature_sets:

        print(f"Discovering feature: {feature}")

        feature_checkpoint_path = Path(
            "checkpoints",
            feature_sets[feature],
        )

        checkpoint_path = Path(
            feature_checkpoint_path,
            checkpoint_name,
        )

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
            interpret=True,
            interpret_features_indices=relevant_feature_indices,
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

        output_data_with_dev = compute_interpret_distance_deviation(
            model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_cov=train_cov,
            test_cov=test_cov,
            latent_dim=hyperparameters.get("latent_dim"),
            output_data=output_data,
        )

        ind_dim_dev_U_test_results = get_individual_deviation_p_values(
            output_data_with_dev,
            hyperparameters.get("latent_dim"),
        )

        print(ind_dim_dev_U_test_results)

        # t_test_p_values = welch_t_test_p_values(
        #     output_data=output_data_with_dev,
        # )

        # print(f"p values of feature {feature}: {t_test_p_values}")

    # welch_t_p_values_dict = welch_t_test_p_values(
    #     output_data=output_data_with_dev,
    # )

    # print(f"p values of feature {feature}: {welch_t_p_values_dict}")

    # kruskal_wallis_test_p_values_dict = kruskal_wallis_test_p_values(
    #     low_symp_distance,
    #     inter_distance,
    #     exter_distance,
    #     high_distance,
    # )

    # plot_histograms(
    #     FEATURE_NAMES_MAP.get(feature),
    #     output_data_with_dev,
    # )

    # plot_boxplots(
    #     FEATURE_NAMES_MAP.get(feature),
    #     output_data_with_dev,
    # )

    # plot_density_distributions(
    #     FEATURE_NAMES_MAP.get(feature),
    #     low_symp_distance,
    #     inter_distance,
    #     exter_distance,
    #     high_distance,
    # )

    # print(f"p values of feature {feature}: {welch_t_p_values_dict}")

    # print(
    #     f"kruskal wallis p values of feature {feature}: {kruskal_wallis_test_p_values_dict}"
    # )


if __name__ == "__main__":

    discover()
