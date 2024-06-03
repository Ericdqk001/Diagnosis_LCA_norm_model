import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO correlation tests between distance values and latent class membership conditional
# probabilities


def latent_deviations_mahalanobis_across(cohort, train):
    dists = calc_robust_mahalanobis_distance(cohort[0], train[0])
    return dists


def calc_robust_mahalanobis_distance(values, train_values):
    robust_cov = MinCovDet(random_state=42).fit(train_values)
    mahal_robust_cov = robust_cov.mahalanobis(values)
    return mahal_robust_cov


def latent_deviation(mu_train, mu_sample, var_sample):
    var = np.var(mu_train, axis=0)
    return (
        np.sum(
            np.abs(mu_sample - np.mean(mu_train, axis=0)) / np.sqrt(var + var_sample),
            axis=1,
        )
        / mu_sample.shape[1]
    )


def separate_latent_deviation(mu_train, mu_sample, var_sample):
    var = np.var(mu_train, axis=0)
    return (mu_sample - np.mean(mu_train, axis=0)) / np.sqrt(var + var_sample)


def U_test_p_values(output_data):
    """Perform Mann-Whitney U test between control distance and each test distance group."""
    # Initialize lists to store results for DataFrame
    cohorts = []
    u_stats = []
    p_values = []

    # Extract mahalanobis distances for each group
    control_distance = output_data["mahalanobis_distance"][
        output_data["low_symp_test_subs"] == 1
    ].values
    inter_test_distance = output_data["mahalanobis_distance"][
        output_data["inter_test_subs"] == 1
    ].values
    exter_test_distance = output_data["mahalanobis_distance"][
        output_data["exter_test_subs"] == 1
    ].values
    high_test_distance = output_data["mahalanobis_distance"][
        output_data["high_test_subs"] == 1
    ].values

    # Define the groups to test against control
    test_groups = {
        "inter_test": inter_test_distance,
        "exter_test": exter_test_distance,
        "high_test": high_test_distance,
    }

    # Perform Mann-Whitney U test for each group
    for group, distances in test_groups.items():
        u_stat, p_value = stats.mannwhitneyu(
            control_distance, distances, alternative="two-sided"
        )
        cohorts.append(group)
        u_stats.append(u_stat)
        p_values.append(p_value)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(
        {"Cohort": cohorts, "U_statistic": u_stats, "P_value": p_values}
    )

    return results_df


def test_assumptions_for_m_distances(feature, output_data):
    """Check assumptions of normality and equal variance for different test groups
    regarding the mahalanobis distance distributions.

    Results, all failed the normality test, one failed the equal variance test,
    justifying the use of non-parametric tests.
    """
    # Extract Mahalanobis distances for each group
    groups = {
        "control": output_data["mahalanobis_distance"][
            output_data["low_symp_test_subs"] == 1
        ].values,
        "inter_test": output_data["mahalanobis_distance"][
            output_data["inter_test_subs"] == 1
        ].values,
        "exter_test": output_data["mahalanobis_distance"][
            output_data["exter_test_subs"] == 1
        ].values,
        "high_test": output_data["mahalanobis_distance"][
            output_data["high_test_subs"] == 1
        ].values,
    }

    # Initialize list to store results for DataFrame
    results = []

    # Test for normality using the Shapiro-Wilk test
    for group_name, distances in groups.items():
        shapiro_stat, shapiro_p = stats.shapiro(distances)
        results.append(
            {
                "Group": group_name,
                "Test": "Shapiro-Wilk",
                "Statistic": shapiro_stat,
                "P-Value": shapiro_p,
                "Feature": feature,
            }
        )

    # Test for equal variances using Levene's test
    levene_stat, levene_p = stats.levene(
        groups["control"],
        groups["inter_test"],
        groups["exter_test"],
        groups["high_test"],
        center="median",  # Recommended when distributions are not symmetrical
    )
    results.append(
        {
            "Group": "All",
            "Test": "Levene",
            "Statistic": levene_stat,
            "P-Value": levene_p,
            "Feature": feature,
        }
    )

    # Convert results list to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def one_hot_encode_covariate(
    data,
    covariate,
    subjects,
) -> np.ndarray:
    """Return one hot encoded covariate for the given subjects as required by the cVAE model."""
    covariate_data = data.loc[
        subjects,
        [covariate],
    ]

    covariate_data[covariate] = pd.Categorical(covariate_data[covariate])

    category_codes = covariate_data[covariate].cat.codes

    num_categories = len(covariate_data[covariate].cat.categories)

    one_hot_encoded_covariate = np.eye(num_categories)[category_codes]

    return one_hot_encoded_covariate


def prepare_inputs_cVAE(
    feature,
    brain_features_path=None,
    cbcl_path=None,
    brain_features_of_interest_path=None,
    data_splits_path=None,
    if_low_entropy=False,
    interpret=False,
    interpret_features_indices=None,
):
    """Prepare the inputs for the compute_mahalanobis_distance_cVAE function below.

    Prepares the data to compute distance between the training latent distribution and
    the latent values of the test subjects in different groups.
    """
    brain_features = pd.read_csv(
        brain_features_path,
        index_col=0,
        low_memory=False,
    )

    # Drop the predicted class column because of overlap in the brain features dataframe
    cbcl_lca = pd.read_csv(
        cbcl_path,
        index_col=0,
        low_memory=False,
    ).drop(columns=["predicted_class"])

    data = brain_features.join(cbcl_lca, how="inner")

    with open(brain_features_of_interest_path, "r") as f:
        brain_features_of_interest = json.load(f)

    with open(data_splits_path, "r") as f:
        data_splits = json.load(f)

    features = brain_features_of_interest[feature]

    if interpret == True:
        features = [features[i] for i in interpret_features_indices]

    if "cortical" in feature:
        modality_data_splits = data_splits["structural"]
    else:
        modality_data_splits = data_splits["functional"]

    train_subs = modality_data_splits["train"]
    test_subs = modality_data_splits["total_test"]
    low_symp_test_subs = modality_data_splits["low_symptom_test"]
    inter_test_subs = modality_data_splits["internalising_test"]
    exter_test_subs = modality_data_splits["externalising_test"]
    high_test_subs = modality_data_splits["high_symptom_test"]

    # if (
    #     test_subs
    #     == low_symp_test_subs + inter_test_subs + exter_test_subs + high_test_subs
    # ):
    #     print("Test subjects list is correctly combined.")

    if if_low_entropy == True:
        # Remove subjects with high entropy
        high_entropy_subs_path = Path(
            "data",
            "LCA",
            "subjects_with_high_entropy.csv",
        )

        high_entropy_subs = pd.read_csv(
            high_entropy_subs_path,
            low_memory=False,
        )["subject"].tolist()

        train_subs = [sub for sub in train_subs if sub not in high_entropy_subs]

        test_subs = [sub for sub in test_subs if sub not in high_entropy_subs]

        low_symp_test_subs = [
            sub for sub in low_symp_test_subs if sub not in high_entropy_subs
        ]

        inter_test_subs = [
            sub for sub in inter_test_subs if sub not in high_entropy_subs
        ]

        exter_test_subs = [
            sub for sub in exter_test_subs if sub not in high_entropy_subs
        ]

        high_test_subs = [sub for sub in high_test_subs if sub not in high_entropy_subs]

        # Tested if test_subs aligns with the combined test subjects list (Works)
        # if (
        #     test_subs
        #     == low_symp_test_subs + inter_test_subs + exter_test_subs + high_test_subs
        # ):
        #     print(
        #         "Test subjects list is correctly combined after removing high entropy."
        #     )

    scaler = StandardScaler()

    train_dataset = data.loc[
        train_subs,
        features,
    ]

    train_dataset_scaled = scaler.fit_transform(train_dataset)

    # Convert the numpy array back to a DataFrame
    train_dataset = pd.DataFrame(
        train_dataset_scaled, index=train_dataset.index, columns=train_dataset.columns
    )

    train_cov = one_hot_encode_covariate(
        data,
        "demo_sex_v2",
        train_subs,
    )

    test_dataset = data.loc[
        test_subs,
        features,
    ]

    test_dataset_scaled = scaler.transform(test_dataset)

    test_dataset = pd.DataFrame(
        test_dataset_scaled,
        index=test_dataset.index,
        columns=test_dataset.columns,
    )

    test_cov = one_hot_encode_covariate(
        data,
        "demo_sex_v2",
        test_subs,
    )

    input_dim = train_dataset.shape[1]

    c_dim = train_cov.shape[1]

    output_data = {
        "low_symp_test_subs": [
            1 if sub in low_symp_test_subs else 0 for sub in test_subs
        ],
        "inter_test_subs": [1 if sub in inter_test_subs else 0 for sub in test_subs],
        "exter_test_subs": [1 if sub in exter_test_subs else 0 for sub in test_subs],
        "high_test_subs": [1 if sub in high_test_subs else 0 for sub in test_subs],
    }

    # Create the DataFrame with indexes set by test_set_subs
    output_data = pd.DataFrame(output_data, index=test_subs)

    return (
        train_dataset,
        test_dataset,
        train_cov,
        test_cov,
        input_dim,
        c_dim,
        output_data,
    )


def compute_distance_deviation_cVAE(
    model,
    train_dataset=None,
    test_dataset=None,
    train_cov=None,
    test_cov=None,
    latent_dim=None,
    output_data=None,
) -> pd.DataFrame:
    """Computes the mahalanobis distance of test samples from the distribution of the
    training samples.
    """
    train_latent, _ = model.pred_latent(
        train_dataset,
        train_cov,
        DEVICE,
    )

    test_latent, test_var = model.pred_latent(
        test_dataset,
        test_cov,
        DEVICE,
    )

    test_distance = latent_deviations_mahalanobis_across(
        [test_latent],
        [train_latent],
    )

    output_data["mahalanobis_distance"] = test_distance

    output_data["latent_deviation"] = latent_deviation(
        train_latent, test_latent, test_var
    )

    individual_deviation = separate_latent_deviation(
        train_latent, test_latent, test_var
    )
    for i in range(latent_dim):
        output_data["latent_deviation_{0}".format(i)] = individual_deviation[:, i]

    return output_data


def compute_interpret_distance_deviation_cVAE(
    model,
    train_dataset=None,
    test_dataset=None,
    train_cov=None,
    test_cov=None,
    latent_dim=None,
    output_data=None,
) -> pd.DataFrame:
    """Computes the mahalanobis distance of test samples from the distribution of the
    training samples.
    """
    train_latent, _ = model.pred_latent(
        train_dataset,
        train_cov,
        DEVICE,
    )

    test_latent, test_var = model.pred_latent(
        test_dataset,
        test_cov,
        DEVICE,
    )

    test_distance = latent_deviations_mahalanobis_across(
        [test_latent],
        [train_latent],
    )

    output_data["mahalanobis_distance"] = test_distance

    individual_deviation = separate_latent_deviation(
        train_latent, test_latent, test_var
    )
    for i in range(latent_dim):
        output_data["latent_deviation_{0}".format(i)] = individual_deviation[:, i]

    return output_data


def get_individual_deviation_p_values(
    feature,
    output_data,
    latent_dim,
    clinical_cohorts=["inter_test_subs", "exter_test_subs", "high_test_subs"],
):
    # Lists to store the results
    results = []
    normality_results = []
    variance_results = []

    for clinical_cohort in clinical_cohorts:
        for i in range(latent_dim):
            # Extract the deviations for normative and clinical groups
            normative_deviation = output_data[f"latent_deviation_{i}"][
                output_data["low_symp_test_subs"] == 1
            ]
            clinical_deviation = output_data[f"latent_deviation_{i}"][
                output_data[clinical_cohort] == 1
            ]

            # Normality test (Shapiro-Wilk)
            shapiro_norm = stats.shapiro(normative_deviation)
            shapiro_clin = stats.shapiro(clinical_deviation)
            normality_results.append(
                {
                    "clinical_cohort": clinical_cohort,
                    "latent_dim": i,
                    "normative_normality_p": shapiro_norm.pvalue,
                    "clinical_normality_p": shapiro_clin.pvalue,
                    "feature": feature,
                }
            )

            # Test for equal variances (Levene's test)
            levene_stat, levene_p = stats.levene(
                normative_deviation, clinical_deviation
            )
            variance_results.append(
                {
                    "clinical_cohort": clinical_cohort,
                    "latent_dim": i,
                    "levene_stat": levene_stat,
                    "levene_p": levene_p,
                    "feature": feature,
                }
            )

            # Mann-Whitney U test for individual deviation
            u_stat, p_value = stats.mannwhitneyu(
                normative_deviation, clinical_deviation, alternative="two-sided"
            )

            # Determine the direction of the difference
            direction = (
                "normative > clinical"
                if normative_deviation.median() > clinical_deviation.median()
                else "normative < clinical"
            )

            # Append the results to the list
            results.append(
                {
                    "clinical_cohort": clinical_cohort,
                    "latent_dim": i,
                    "u_stat": u_stat,
                    "p_value": p_value,
                    "direction": direction,
                    "feature": feature,
                }
            )

    # Create DataFrames from the results list
    results_df = pd.DataFrame(results)
    normality_df = pd.DataFrame(normality_results)
    variance_df = pd.DataFrame(variance_results)

    # Return all results as a tuple of DataFrames
    return results_df, normality_df, variance_df


def identify_extreme_deviation(
    output_data,
    alpha=0.001,
    latent_dim=10,
):
    # Calculate the critical value based on the chi-squared distribution
    critical_value = stats.chi2.ppf(1 - alpha, latent_dim)

    print("Critical value for chi-squared test:")
    print(critical_value)

    # Initialize a dictionary to hold results
    results = {"cohort": [], "proportion_extreme_deviation": []}

    # Cohorts list
    cohorts = [
        "low_symp_test_subs",
        "inter_test_subs",
        "exter_test_subs",
        "high_test_subs",
    ]

    # Analyze each cohort
    for cohort in cohorts:
        # Filter data for the current cohort
        cohort_data = output_data[output_data[cohort] == 1]["mahalanobis_distance"]

        # Calculate the proportion of distances that are extreme deviations
        proportion_extreme = (cohort_data > critical_value).mean()

        # Append the results to the dictionary
        results["cohort"].append(cohort)
        results["proportion_extreme_deviation"].append(proportion_extreme)

    # Convert results dictionary to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def test_correlate_distance_symptom_severity(output_data):
    data_path = Path(
        "data",
        "raw_data",
        "core",
        "mental-health",
        "mh_p_cbcl.csv",
    )

    # cbcl_t_vars_path = Path(
    #     "data",
    #     "var_dict",
    #     "cbcl_8_dim_t.csv",
    # )

    # Load CBCL scores and variable names
    cbcl = pd.read_csv(data_path, index_col=0, low_memory=False)

    # cbcl_t_vars_df = pd.read_csv(cbcl_t_vars_path)
    # cbcl_t_vars = cbcl_t_vars_df["var_name"].tolist()

    sum_syndrome = [
        "cbcl_scr_syn_internal_t",
        "cbcl_scr_syn_external_t",
        "cbcl_scr_syn_totprob_t",
    ]

    baseline_cbcl = cbcl[cbcl["eventname"] == "baseline_year_1_arm_1"]

    # Filter columns with t variables
    filtered_cbcl = baseline_cbcl[sum_syndrome].dropna()

    # Merge datasets
    inter_test_data = output_data[output_data["inter_test_subs"] == 1].join(
        filtered_cbcl, how="inner"
    )
    exter_test_data = output_data[output_data["exter_test_subs"] == 1].join(
        filtered_cbcl, how="inner"
    )
    high_test_data = output_data[output_data["high_test_subs"] == 1].join(
        filtered_cbcl, how="inner"
    )

    # Prepare to store results
    results = []

    # Calculate correlation for each cohort
    cohorts = {
        "Internalizing": inter_test_data,
        "Externalizing": exter_test_data,
        "High Symptom": high_test_data,
    }

    for cohort_name, cohort_data in cohorts.items():
        for syndrome in sum_syndrome:
            # Compute the Spearman correlation and the p-value
            correlation, p_value = stats.spearmanr(
                cohort_data["mahalanobis_distance"], cohort_data[syndrome]
            )
            results.append(
                {
                    "Cohort": cohort_name,
                    "Syndrome": syndrome,
                    "Correlation": correlation,
                    "P-Value": p_value,
                }
            )

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df


# Interpretation
# If the squared Mahalanobis distance of a point exceeds this critical value,
# it can be considered an extreme outlier at the 1% significance level.


# def reconstruction_deviation(x, x_pred):
#     return x_pred - x


# def get_mean_std_recon_deviation(x_norm, x_pred_norm):
#     recon_deviation = reconstruction_deviation(x_norm, x_pred_norm)
#     mean_recon_deviation = np.mean(recon_deviation)
#     std_recon_deviation = np.std(recon_deviation)
#     return mean_recon_deviation, std_recon_deviation


# def recon_deviation_sum(x, x_pred):
#     feat_dim = x.shape[1]
#     dev = np.sum(np.sqrt((x - x_pred) ** 2), axis=1) / feat_dim
#     return dev


# def get_normalised_recon_deviation(
#     x, x_pred, mean_recon_deviation, std_recon_deviation
# ):
#     recon_deviation = reconstruction_deviation(x, x_pred)
#     normalised_recon_deviation = (
#         recon_deviation - mean_recon_deviation
#     ) / std_recon_deviation
#     return normalised_recon_deviation
