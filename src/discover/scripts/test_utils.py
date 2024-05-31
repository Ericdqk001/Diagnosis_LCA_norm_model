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


def test_assumptions_for_m_distances(output_data):
    """Check assumptions of normality and equal variance for different test groups
    regarding the mahalanobis distance distributions.

    Results suggests that most distribution does not satisfy the assumptions of
    normality, one did not satisfy the assumption of equal variance.
    ."""
    # Initialize dictionary to store results
    results = {}

    # Extract Mahalanobis distances for each group
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

    # Test for normality using the Shapiro-Wilk test
    results["normality"] = {
        "control": stats.shapiro(control_distance),
        "inter_test": stats.shapiro(inter_test_distance),
        "exter_test": stats.shapiro(exter_test_distance),
        "high_test": stats.shapiro(high_test_distance),
    }

    # Test for equal variances using Levene's test
    # Note: Welch's test does not assume equal variances.
    results["levenes_test"] = stats.levene(
        control_distance,
        inter_test_distance,
        exter_test_distance,
        high_test_distance,
        center="median",  # Recommended when distributions are not symmetrical
    )

    return results


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

    data = {
        "low_symp_test_subs": [
            1 if sub in low_symp_test_subs else 0 for sub in test_subs
        ],
        "inter_test_subs": [1 if sub in inter_test_subs else 0 for sub in test_subs],
        "exter_test_subs": [1 if sub in exter_test_subs else 0 for sub in test_subs],
        "high_test_subs": [1 if sub in high_test_subs else 0 for sub in test_subs],
    }

    # Create the DataFrame with indexes set by test_set_subs
    output_data = pd.DataFrame(data, index=test_subs)

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
                }
            )

    # Create DataFrames from the results list
    results_df = pd.DataFrame(results)
    normality_df = pd.DataFrame(normality_results)
    variance_df = pd.DataFrame(variance_results)

    # Return all results as a tuple of DataFrames
    return results_df, normality_df, variance_df


def identify_extreme_deviation():
    pass


# Parameters
dimensions = 10  # Number of dimensions
alpha = 0.001  # Significance level

# Critical value for chi-squared distribution at the given alpha
critical_value = stats.chi2.ppf(1 - alpha, dimensions)
print("Critical value for extreme deviation:", critical_value)

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


# def welch_t_test_p_values(
#     output_data,
# ) -> dict:
#     """Perform Welch's t-test between control distance and each test distance group."""
#     p_values = {}

#     control_distance = output_data["mahalanobis_distance"][
#         output_data["low_symp_test_subs"] == 1
#     ].values

#     inter_test_distance = output_data["mahalanobis_distance"][
#         output_data["inter_test_subs"] == 1
#     ].values

#     exter_test_distance = output_data["mahalanobis_distance"][
#         output_data["exter_test_subs"] == 1
#     ].values

#     high_test_distance = output_data["mahalanobis_distance"][
#         output_data["high_test_subs"] == 1
#     ].values

#     # Welch's t-test for inter_test_distance
#     t_stat_inter, p_value_inter = stats.ttest_ind(
#         control_distance, inter_test_distance, equal_var=False
#     )
#     p_values["inter_test"] = p_value_inter

#     # Welch's t-test for exter_test_distance
#     t_stat_exter, p_value_exter = stats.ttest_ind(
#         control_distance, exter_test_distance, equal_var=False
#     )
#     p_values["exter_test"] = p_value_exter

#     # Welch's t-test for high_test_distance
#     t_stat_high, p_value_high = stats.ttest_ind(
#         control_distance, high_test_distance, equal_var=False
#     )
#     p_values["high_test"] = p_value_high

#     return p_values
