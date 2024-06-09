from pathlib import Path

import pandas as pd
from discover.scripts.test_utils import U_test_p_values

cVAE_discover_results_path = Path(
    "src",
    "discover",
    "results",
)


output_data_save_path = Path(
    cVAE_discover_results_path,
    "output_data",
)


feature_sets = {
    "t1w_cortical_thickness_rois": "Cortical Thickness",
    # "t1w_cortical_volume_rois": "Cortical Volume",
    # "t1w_cortical_surface_area_rois": "Cortical Surface Area",
}


def discover(num_brain_features=148):

    ind_U_test_results = []

    for feature in feature_sets:

        U_test_results = []

        print(f"Discovering feature: {feature}")

        feature_output_data_save_path = Path(
            output_data_save_path,
            f"{feature}_output_data_with_dev.csv",
        )

        feature_output_data = pd.read_csv(
            feature_output_data_save_path,
            index_col=0,
        )

        for i in range(num_brain_features):

            metric = f"reconstruction_deviation_{i}"

            U_test_result = U_test_p_values(
                feature_output_data,
                metric=metric,
            )

            U_test_result["metric"] = metric

            U_test_results.append(U_test_result)

        print(len(U_test_results))

        U_test_results_df = pd.concat(U_test_results)

        print(U_test_results_df)

        U_test_results_df["Feature"] = feature

        ind_U_test_results.append(U_test_results_df)

    feature_U_test_results_df = pd.concat(ind_U_test_results)

    ind_brain_region_U_test_results_path = Path(
        cVAE_discover_results_path,
        "ind_brain_region_U_test_results",
    )

    if not ind_brain_region_U_test_results_path.exists():
        ind_brain_region_U_test_results_path.mkdir(parents=True)

    # feature_U_test_results_df.to_csv(
    #     Path(
    #         ind_brain_region_U_test_results_path,
    #         "ind_brain_region_U_test_results.csv",
    #     )
    # )


if __name__ == "__main__":
    discover()


### NOTE All individual reconstruction deviation for all brain regions and groups failed
# the normality test.
