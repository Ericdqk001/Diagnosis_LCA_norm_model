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


def welch_t_test_p_values(
    output_data,
) -> dict:
    """Perform Welch's t-test between control distance and each test distance group."""
    p_values = {}

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

    # Welch's t-test for inter_test_distance
    t_stat_inter, p_value_inter = stats.ttest_ind(
        control_distance, inter_test_distance, equal_var=False
    )
    p_values["inter_test"] = p_value_inter

    # Welch's t-test for exter_test_distance
    t_stat_exter, p_value_exter = stats.ttest_ind(
        control_distance, exter_test_distance, equal_var=False
    )
    p_values["exter_test"] = p_value_exter

    # Welch's t-test for high_test_distance
    t_stat_high, p_value_high = stats.ttest_ind(
        control_distance, high_test_distance, equal_var=False
    )
    p_values["high_test"] = p_value_high

    return p_values


def kruskal_wallis_test_p_values(
    control_distance: np.ndarray,
    inter_test_distance: np.ndarray,
    exter_test_distance: np.ndarray,
    high_test_distance: np.ndarray,
) -> dict:
    """Perform Kruskal-Wallis test between control distance and each test distance group."""
    p_values = {}

    # Combine all distances into a list of samples
    all_distances = [
        control_distance,
        inter_test_distance,
        exter_test_distance,
        high_test_distance,
    ]

    # Perform Kruskal-Wallis test
    h_stat, p_value = stats.kruskal(*all_distances)

    # Store p-value from Kruskal-Wallis test
    p_values["kruskal_test"] = p_value

    return p_values


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
    if_no_psych_dx=False,
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

    if if_no_psych_dx == True:

        psych_dx_path = Path(
            "data",
            "liza_data",
            "all_psych_dx_r5.csv",
        )

        psych_dx = pd.read_csv(
            psych_dx_path,
            index_col=0,
            low_memory=False,
        )

        control_subs = psych_dx[psych_dx["psych_dx"] == "control"].index.tolist()

        train_subs = [sub for sub in train_subs if sub in control_subs]

        low_symp_test_subs = [sub for sub in low_symp_test_subs if sub in control_subs]

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


def get_individual_deviation_p_values(
    output_data,
    latent_dim,
    clinical_cohorts=["inter_test_subs", "exter_test_subs", "high_test_subs"],
):
    # Lists to store the results
    results = []

    for clinical_cohort in clinical_cohorts:
        for i in range(latent_dim):
            # Mann-Whitney U test for individual deviation
            normative_deviation = output_data["latent_deviation_{0}".format(i)][
                output_data["low_symp_test_subs"] == 1
            ]

            clinical_deviation = output_data["latent_deviation_{0}".format(i)][
                output_data[clinical_cohort] == 1
            ]

            u_stat, p_value = stats.mannwhitneyu(
                normative_deviation,
                clinical_deviation,
                alternative="two-sided",
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

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)

    return results_df


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
