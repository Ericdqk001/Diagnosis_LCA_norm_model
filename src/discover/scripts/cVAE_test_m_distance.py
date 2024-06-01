from pathlib import Path

import pandas as pd
from discover.scripts.plot_utils import (
    plot_boxplots,
    plot_density_distributions,
    plot_histograms,
)
from discover.scripts.test_utils import U_test_p_values, identify_extreme_deviation

cVAE_discover_results_path = Path(
    "src",
    "discover",
    "results",
)


output_data_save_path = Path(
    cVAE_discover_results_path,
    "out_put_data",
)


feature_sets = {
    "t1w_cortical_thickness_rois": "cortical_thickness",
    "t1w_cortical_volume_rois": "cortical_volume",
    "t1w_cortical_surface_area_rois": "cortical_surface_area",
    "gordon_net_subcor_limbic_no_dup": "rsfmri",
}


def discover():

    U_test_results = []

    extreme_deviation_results = []

    for feature in feature_sets:

        print(f"Discovering feature: {feature}")

        feature_output_data_save_path = Path(
            output_data_save_path,
            f"{feature}_output_data_with_dev.csv",
        )

        feature_output_data = pd.read_csv(feature_output_data_save_path)

        plot_histograms(
            feature=feature,
            output_data=feature_output_data,
        )

        plot_boxplots(
            feature=feature,
            output_data=feature_output_data,
        )

        plot_density_distributions(
            feature=feature,
            output_data=feature_output_data,
        )

        U_test_result = U_test_p_values(
            output_data=feature_output_data,
        )

        U_test_result["Feature"] = feature_sets[feature]

        U_test_results.append(
            U_test_result,
        )

        cohort_extreme_dev_prop = identify_extreme_deviation(
            output_data=feature_output_data,
        )

        cohort_extreme_dev_prop["Feature"] = feature_sets[feature]

        extreme_deviation_results.append(
            cohort_extreme_dev_prop,
        )

    U_test_results_df = pd.concat(U_test_results)

    U_test_save_results_path = Path(
        cVAE_discover_results_path,
        "mahalabonis_U_test_results",
    )

    if not U_test_save_results_path.exists():
        U_test_save_results_path.mkdir(parents=True)

    U_test_results_df.to_csv(
        Path(
            U_test_save_results_path,
            "U_test_results.csv",
        ),
        index=False,
    )

    extreme_deviation_results_df = pd.concat(extreme_deviation_results)

    extreme_deviation_save_results_path = Path(
        cVAE_discover_results_path,
        "extreme_deviation_results",
    )

    if not extreme_deviation_save_results_path.exists():
        extreme_deviation_save_results_path.mkdir(parents=True)

    extreme_deviation_results_df.to_csv(
        Path(
            extreme_deviation_save_results_path,
            "extreme_deviation_results.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    discover()
