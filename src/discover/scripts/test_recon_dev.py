from pathlib import Path

import pandas as pd
from discover.scripts.plot_utils import (
    plot_boxplots,
    plot_histograms,
)
from discover.scripts.test_utils import (
    U_test_p_values,
    test_assumptions_for_u_test,
    test_correlate_distance_symptom_severity,
)

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

    U_test_results = []

    assumption_test_results = []

    # extreme_deviation_results = []

    correlation_results = []

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

        plot_boxplots(
            feature=feature_sets[feature],
            output_data=feature_output_data,
            metric="reconstruction_deviation",
        )

        # Test assumptions of normality and homogeneity of variance
        assumption_test_result = test_assumptions_for_u_test(
            feature=feature,
            output_data=feature_output_data,
            metric="reconstruction_deviation",
        )

        assumption_test_results.append(
            assumption_test_result,
        )

        U_test_result = U_test_p_values(
            output_data=feature_output_data,
            metric="reconstruction_deviation",
        )

        plot_histograms(
            feature=feature_sets[feature],
            output_data=feature_output_data,
            p_values_df=U_test_result,
            metric="reconstruction_deviation",
        )

        U_test_result["Feature"] = feature_sets[feature]

        U_test_results.append(
            U_test_result,
        )

        correlation_result = test_correlate_distance_symptom_severity(
            feature=feature_sets[feature],
            output_data=feature_output_data,
            metric="reconstruction_deviation",
        )

        correlation_result["Feature"] = feature_sets[feature]

        correlation_results.append(
            correlation_result,
        )

    U_test_results_df = pd.concat(U_test_results)

    assumption_test_results_df = pd.concat(assumption_test_results)

    assumption_test_results_path = Path(
        cVAE_discover_results_path,
        "assumption_test_results",
    )

    if if_low_entropy:
        assumption_test_results_path = Path(
            cVAE_discover_results_path,
            "low_entropy_assumption_test_results",
        )

    if not assumption_test_results_path.exists():
        assumption_test_results_path.mkdir(parents=True)

    assumption_test_results_df.to_csv(
        Path(
            assumption_test_results_path,
            "recon_dev_assumption_test_results.csv",
        ),
        index=False,
    )

    U_test_save_results_path = Path(
        cVAE_discover_results_path,
        "U_test_results",
    )

    if if_low_entropy:
        U_test_save_results_path = Path(
            cVAE_discover_results_path,
            "low_entropy_U_test_results",
        )

    if not U_test_save_results_path.exists():
        U_test_save_results_path.mkdir(parents=True)

    U_test_results_df.to_csv(
        Path(
            U_test_save_results_path,
            "recon_dev_U_test_results.csv",
        ),
        index=False,
    )

    correlation_results_df = pd.concat(correlation_results)

    correlation_results_save_path = Path(
        cVAE_discover_results_path,
        "correlation_results",
    )

    if if_low_entropy:
        correlation_results_save_path = Path(
            cVAE_discover_results_path,
            "low_entropy_correlation_results",
        )

    if not correlation_results_save_path.exists():
        correlation_results_save_path.mkdir(parents=True)

    correlation_results_df.to_csv(
        Path(
            correlation_results_save_path,
            "recon_dev_correlation_results.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    discover(if_low_entropy=True)
