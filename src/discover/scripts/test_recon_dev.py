from pathlib import Path

import pandas as pd
from discover.scripts.plot_utils import (
    plot_boxplots,
    plot_correlations,
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
    # "t1w_cortical_thickness_rois": "Cortical Thickness",
    # "t1w_cortical_volume_rois": "Cortical Volume",
    # "t1w_cortical_surface_area_rois": "Cortical Surface Area",
    "gordon_net_subcor_limbic_no_dup": "Functional Connectivity",
}


def discover(
    low_entropy: bool = False,
    metric: str = "reconstruction_deviation",
    results_path: Path = cVAE_discover_results_path,
):

    if low_entropy:
        results_path = Path(
            results_path,
            "low_entropy",
        )

    output_data_save_path = Path(
        cVAE_discover_results_path,
        "output_data",
    )

    if low_entropy:

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
            metric=metric,
        )

        # Test assumptions of normality and homogeneity of variance
        assumption_test_result = test_assumptions_for_u_test(
            feature=feature,
            output_data=feature_output_data,
            metric=metric,
        )

        assumption_test_results.append(
            assumption_test_result,
        )

        U_test_result = U_test_p_values(
            output_data=feature_output_data,
            metric=metric,
        )

        plot_histograms(
            feature=feature_sets[feature],
            output_data=feature_output_data,
            p_values_df=U_test_result,
            metric=metric,
        )

        plot_correlations(
            feature=feature_sets[feature],
            output_data=feature_output_data,
            metric=metric,
        )

        U_test_result["Feature"] = feature_sets[feature]

        U_test_results.append(
            U_test_result,
        )

        correlation_result = test_correlate_distance_symptom_severity(
            feature=feature_sets[feature],
            output_data=feature_output_data,
            metric=metric,
        )

        correlation_result["Feature"] = feature_sets[feature]

        correlation_results.append(
            correlation_result,
        )

    U_test_results_df = pd.concat(U_test_results)

    assumption_test_results_df = pd.concat(assumption_test_results)

    assumption_test_results_path = Path(
        results_path,
        "assumption_test_results",
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
        results_path,
        "U_test_results",
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
        results_path,
        "correlation_results",
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
    discover(low_entropy=False, metric="reconstruction_deviation")
