import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from modelling.load.load import MyDataset_labels
from modelling.models.cVAE import cVAE
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description="Tune cVAE")


# python src/modelling/bootstrap/cVAE_bootstrap.py --data_path "data/processed_data" --feature_type "cortical_thickness" --project_title "cVAE_rsfmri_final_train" --batch_size 256 --learning_rate 0.0005 --latent_dim 10 --hidden_dim "40" --bootstrap_num 10

# python src/modelling/bootstrap/cVAE_bootstrap.py --data_path "data/processed_data" --feature_type "cortical_volume" --project_title "cVAE_rsfmri_final_train" --batch_size 256 --learning_rate 0.001 --latent_dim 10 --hidden_dim "30-30"

# python src/modelling/bootstrap/cVAE_bootstrap.py --data_path "data/processed_data" --feature_type "cortical_surface_area" --project_title "cVAE_rsfmri_final_train" --batch_size 256 --learning_rate 0.0005 --latent_dim 10 --hidden_dim "30-30"


def int_parse_list(arg_value):
    return [int(x) for x in arg_value.split("-")]


parser.add_argument(
    "--data_path",
    type=str,
    default="",
    help="Path to processed data",
)
parser.add_argument(
    "--feature_type",
    type=str,
    default="",
    help="Brain features of interest, options: 'cortical_thickness, cortical_volume, rsfmri'",
)
parser.add_argument(
    "--project_title",
    type=str,
    default="Train cVAE",
    help="Title of the project for weights and biases",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for training, e.g., '64-128-256', which will be parsed into [64, 128, 256]",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.01,
    help="Learning rate for the optimizer, e.g., '0.005-0.001-0.0005-0.0001', which will be parsed into [0.005, 0.001, 0.0005, 0.0001]",
)
parser.add_argument(
    "--latent_dim",
    type=int,
    default=5,
    help="Dimensions of the latent space, e.g., '10-11-12', which will be parsed into [10, 11, 12]",
)
parser.add_argument(
    "--hidden_dim",
    type=int_parse_list,
    default=[30],
    help="Pass dimensions of multiple hidden layers, use ';' to separate layers and '-' to separate dimensions within layers, e.g., '30-30;40-40-40'",
)
parser.add_argument(
    "--bootstrap_num",
    type=int,
    default=1000,
    help="Number of bootstrap iterations",
)

args = parser.parse_args()
processed_data_path = Path(args.data_path)
feature_type = args.feature_type

feature_sets_map = {
    "cortical_thickness": "t1w_cortical_thickness_rois",
    "cortical_volume": "t1w_cortical_volume_rois",
    "cortical_surface_area": "t1w_cortical_surface_area_rois",
    "rsfmri": "gordon_net_subcor_limbic_no_dup",
}


brain_features_of_interest_path = Path(
    processed_data_path,
    "brain_features_of_interest.json",
)

with open(brain_features_of_interest_path, "r") as f:
    brain_features_of_interest = json.load(f)

FEATURES = brain_features_of_interest[feature_sets_map[feature_type]]


TRAIN_DATA_PATH = Path(
    processed_data_path,
    "all_brain_features_resid_exc_sex.csv",
)

# CBCL data path for getting the outputs
CBCL_DATA_PATH = Path(
    processed_data_path,
    "mh_p_cbcl.csv",
)

CHECKPOINT_PATH = Path(
    "checkpoints",
    feature_type,
)

if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.mkdir(parents=True)

data_splits_path = Path(
    processed_data_path,
    "data_splits_with_clinical_val.json",
)

with open(data_splits_path, "r") as f:
    data_splits = json.load(f)

if "cortical" in feature_type:
    DATA_SPLITS = data_splits["structural"]

else:
    DATA_SPLITS = data_splits["functional"]


TRAIN_SUBS = DATA_SPLITS["train"]
VAL_SUBS = DATA_SPLITS["val"]


def build_model(
    config,
    input_dim,
    c_dim,
):
    # Set random seed for CPU
    torch.manual_seed(123)

    # Set random seed for CUDA (GPU) if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    # Initialize the model
    model = cVAE(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        c_dim=c_dim,
        learning_rate=config["learning_rate"],
        non_linear=True,
    ).to(DEVICE)

    if isinstance(model, nn.Module):
        first_layer_params = list(model.parameters())[0]
        print("Parameter values of the first layer:", first_layer_params)

    return model


def one_hot_encode_covariate(
    data,
    covariate,
    subjects,
):
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


def validate(model, val_loader, device):
    model.eval()
    total_val_loss = 0.0
    total_recon = 0.0
    total_kl_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            data_curr = batch[0].to(device)
            cov = batch[1].to(device)
            fwd_rtn = model.forward(data_curr, cov)
            val_loss = model.loss_function(data_curr, fwd_rtn)
            batch_val_loss = val_loss["total"].item()
            total_val_loss += batch_val_loss
            total_recon += val_loss["ll"].item()
            total_kl_loss += val_loss["kl"].item()

    mean_val_loss = total_val_loss / len(val_loader)

    return mean_val_loss


def latent_deviations_mahalanobis_across(cohort, train):
    dists = calc_robust_mahalanobis_distance(cohort[0], train[0])
    return dists


def calc_robust_mahalanobis_distance(values, train_values):

    # Compute the robust covariance matrix
    robust_cov = MinCovDet(random_state=42).fit(train_values)

    # Calculate the Mahalanobis distance using the robust covariance matrix
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


def reconstruction_deviation(x, x_pred):

    dev = np.mean((x - x_pred) ** 2, axis=1)

    return dev


def ind_reconstruction_deviation(x, x_pred):
    dev = (x - x_pred) ** 2

    return dev


def standardise_reconstruction_deviation(output_data):
    control_recon_dev = output_data["reconstruction_deviation"][
        output_data["low_symp_test_subs"] == 1
    ].values

    inter_test_recon_dev = output_data["reconstruction_deviation"][
        output_data["inter_test_subs"] == 1
    ].values

    exter_test_recon_dev = output_data["reconstruction_deviation"][
        output_data["exter_test_subs"] == 1
    ].values

    high_test_recon_dev = output_data["reconstruction_deviation"][
        output_data["high_test_subs"] == 1
    ].values

    mu = np.mean(control_recon_dev)
    sigma = np.std(control_recon_dev)

    control_recon_dev = (control_recon_dev - mu) / sigma

    inter_test_recon_dev = (inter_test_recon_dev - mu) / sigma

    exter_test_recon_dev = (exter_test_recon_dev - mu) / sigma

    high_test_recon_dev = (high_test_recon_dev - mu) / sigma

    standardised_recon_dev = np.concatenate(
        [
            control_recon_dev,
            inter_test_recon_dev,
            exter_test_recon_dev,
            high_test_recon_dev,
        ]
    )

    return standardised_recon_dev


def separate_latent_deviation(mu_train, mu_sample, var_sample):
    var = np.var(mu_train, axis=0)
    return (mu_sample - np.mean(mu_train, axis=0)) / np.sqrt(var + var_sample)


def compute_distance_deviation(
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

    test_prediction = model.pred_recon(
        test_dataset,
        test_cov,
        DEVICE,
    )

    test_distance = latent_deviations_mahalanobis_across(
        [test_latent],
        [train_latent],
    )

    output_data["mahalanobis_distance"] = test_distance

    output_data["reconstruction_deviation"] = reconstruction_deviation(
        test_dataset.to_numpy(),
        test_prediction,
    )

    # Record reconstruction deviation for each brain region

    for i in range(test_prediction.shape[1]):

        output_data[f"reconstruction_deviation_{i}"] = ind_reconstruction_deviation(
            test_dataset.to_numpy()[:, i],
            test_prediction[:, i],
        )

    output_data["standardised_reconstruction_deviation"] = (
        standardise_reconstruction_deviation(output_data)
    )

    output_data["latent_deviation"] = latent_deviation(
        train_latent, test_latent, test_var
    )

    individual_deviation = separate_latent_deviation(
        train_latent, test_latent, test_var
    )
    for i in range(latent_dim):
        output_data["latent_deviation_{0}".format(i)] = individual_deviation[:, i]

    return output_data


def prepare_output(
    data,
    scaler,
):
    test_subs = DATA_SPLITS["total_test"]
    low_symp_test_subs = DATA_SPLITS["low_symptom_test"]
    inter_test_subs = DATA_SPLITS["internalising_test"]
    exter_test_subs = DATA_SPLITS["externalising_test"]
    high_test_subs = DATA_SPLITS["high_symptom_test"]

    test_dataset = data.loc[
        test_subs,
        FEATURES,
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

    cbcl = pd.read_csv(
        CBCL_DATA_PATH,
        index_col=0,
        low_memory=False,
    )

    cbcl_scales = [
        "cbcl_scr_syn_anxdep_t",
        "cbcl_scr_syn_withdep_t",
        "cbcl_scr_syn_somatic_t",
        "cbcl_scr_syn_social_t",
        "cbcl_scr_syn_thought_t",
        "cbcl_scr_syn_attention_t",
        "cbcl_scr_syn_rulebreak_t",
        "cbcl_scr_syn_aggressive_t",
    ]

    sum_syndrome = [
        "cbcl_scr_syn_internal_t",
        "cbcl_scr_syn_external_t",
        "cbcl_scr_syn_totprob_t",
    ]

    all_scales = cbcl_scales + sum_syndrome

    baseline_cbcl = cbcl[cbcl["eventname"] == "baseline_year_1_arm_1"]

    filtered_cbcl = baseline_cbcl[all_scales]

    output_data = output_data.join(filtered_cbcl, how="left")

    output_data["cbcl_sum_score"] = output_data[cbcl_scales].sum(axis=1)

    return output_data, test_dataset, test_cov


def train(
    config,
    model,
    train_loader,
    val_loader,
    tolerance=50,
    output_data=None,
    train_dataset=None,
    train_cov=None,
    test_dataset=None,
    test_cov=None,
):

    model.to(DEVICE)

    best_val_loss = float("inf")

    epochs_no_improve = 0

    for epoch in range(1, config["epochs"] + 1):

        total_loss = 0.0
        total_recon = 0.0
        kl_loss = 0.0

        model.train()

        for batch_idx, batch in enumerate(train_loader):
            data_curr = batch[0].to(DEVICE)
            cov = batch[1].to(DEVICE)
            fwd_rtn = model.forward(data_curr, cov)
            loss = model.loss_function(data_curr, fwd_rtn)
            model.optimizer.zero_grad()
            loss["total"].backward()
            model.optimizer.step()

            total_loss += loss["total"].item()
            total_recon += loss["ll"].item()
            kl_loss += loss["kl"].item()

        val_loss = validate(model, val_loader, DEVICE)

        if val_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_loss

            best_model = model.state_dict()

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= tolerance:

            break

    # Get output
    model.load_state_dict(best_model)

    output_data = compute_distance_deviation(
        model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_cov=train_cov,
        test_cov=test_cov,
        latent_dim=config["latent_dim"],
        output_data=output_data,
    )

    return output_data


def main(config):

    np.random.seed(123)

    data = pd.read_csv(
        Path(TRAIN_DATA_PATH),
        index_col=0,
        low_memory=False,
    )

    combined_output_data = []

    for i in tqdm(range(config["bootstrap_num"]), desc="Bootstrapping"):
        # Bootstrap sampling with replacement
        bootstrap_train_subs = np.random.choice(
            TRAIN_SUBS, size=len(TRAIN_SUBS), replace=True
        )

        train_dataset = data.loc[
            bootstrap_train_subs,
            FEATURES,
        ]

        # Covariate one hot encoding
        encoded_covariate_train = one_hot_encode_covariate(
            data,
            "demo_sex_v2",
            bootstrap_train_subs,
        )

        val_dataset = data.loc[
            VAL_SUBS,
            FEATURES,
        ].to_numpy()

        val_cov = one_hot_encode_covariate(
            data,
            "demo_sex_v2",
            VAL_SUBS,
        )

        c_dim = encoded_covariate_train.shape[1]

        scaler = StandardScaler()

        train_data = scaler.fit_transform(train_dataset)

        output_data, test_dataset, test_cov = prepare_output(data, scaler)

        val_data = scaler.transform(val_dataset)

        # Test if scaler is refreshed
        # print("Scaled val data mean:", np.mean(val_data))

        train_loader = DataLoader(
            MyDataset_labels(train_data, encoded_covariate_train),
            batch_size=config["batch_size"],
            shuffle=True,
        )

        val_loader = DataLoader(
            MyDataset_labels(val_data, val_cov),
            batch_size=config["batch_size"],
            shuffle=False,
        )

        input_dim = train_data.shape[1]

        model = build_model(
            config,
            input_dim,
            c_dim,
        )

        output_data = train(
            config,
            model,
            train_loader,
            val_loader,
            output_data=output_data,
            train_dataset=train_dataset,
            train_cov=encoded_covariate_train,
            test_dataset=test_dataset,
            test_cov=test_cov,
        )

        # Add bootstrap_num column
        output_data["bootstrap_num"] = i

        combined_output_data.append(output_data)

    # Combine all output data from bootstrap iterations
    combined_output_df = pd.concat(combined_output_data, axis=0)

    # Save the combined output data to a CSV file
    combined_output_df.to_csv(
        Path(
            processed_data_path.parent,
            f"{feature_type}_bootstrap_results.csv",
            index=True,
        )
    )


if __name__ == "__main__":

    project_title = args.project_title
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim
    bootstrap_num = args.bootstrap_num

    config = {
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "learning_rate": learning_rate,
        "bootstrap_num": bootstrap_num,
        "epochs": 5000,
    }

    main(config)
