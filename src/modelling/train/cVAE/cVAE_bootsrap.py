import json
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.distributions import Normal
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)

parser = ArgumentParser(description="Tune cVAE")


def parse_list_of_lists(arg_value):
    # Converts a string format like "30-30;40-40-40" into a list of lists [[30, 30], [40, 40, 40]]
    return [[int(y) for y in x.split("-")] for x in arg_value.split(";")]


def float_parse_list(arg_value):
    return [float(x) for x in arg_value.split("-")]


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
    default="Tune cVAE",
    help="Title of the project for weights and biases",
)
parser.add_argument(
    "--batch_size",
    type=int_parse_list,
    default=[64, 128, 256],
    help="Batch size for training, e.g., '64-128-256', which will be parsed into [64, 128, 256]",
)
parser.add_argument(
    "--learning_rate",
    type=float_parse_list,
    default=[
        0.01,
        0.005,
        0.001,
        0.0005,
    ],
    help="Learning rate for the optimizer, e.g., '0.005-0.001-0.0005-0.0001', which will be parsed into [0.005, 0.001, 0.0005, 0.0001]",
)
parser.add_argument(
    "--latent_dim",
    type=int_parse_list,
    default=[10],
    help="Dimensions of the latent space, e.g., '10-11-12', which will be parsed into [10, 11, 12]",
)
parser.add_argument(
    "--hidden_dim",
    type=parse_list_of_lists,
    default=[
        [30],
        [30, 30],
        [40],
        [40, 40],
        [50],
        [50, 50],
    ],
    help="Pass dimensions of multiple hidden layers, use ';' to separate layers and '-' to separate dimensions within layers, e.g., '30-30;40-40-40'",
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

data_splits_path = Path(
    processed_data_path,
    "data_splits_with_clinical_val.json",
)

with open(data_splits_path, "r") as f:
    data_splits = json.load(f)

if "cortical" in feature_type:
    modality_data_split = data_splits["structural"]

else:
    modality_data_split = data_splits["functional"]


TRAIN_SUBS = modality_data_split["train"]


### Model and Dataset class
def compute_ll(x, x_recon):
    return x_recon.log_prob(x).sum(1, keepdims=True).mean(0)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, c_dim, non_linear=False):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_dim
        self.z_dim = hidden_dim[-1]
        self.c_dim = c_dim
        self.non_linear = non_linear
        self.layer_sizes_encoder = [input_dim + c_dim] + self.hidden_dims
        lin_layers = [
            nn.Linear(dim0, dim1, bias=True)
            for dim0, dim1 in zip(
                self.layer_sizes_encoder[:-1], self.layer_sizes_encoder[1:]
            )
        ]

        self.encoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.enc_mean_layer = nn.Linear(
            self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True
        )
        self.enc_logvar_layer = nn.Linear(
            self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True
        )

    def forward(self, x, c):
        c = c.reshape(-1, self.c_dim)
        h1 = torch.cat((x, c), dim=1)
        for it_layer, layer in enumerate(self.encoder_layers):
            h1 = layer(h1)
            if self.non_linear:
                h1 = F.relu(h1)

        mu = self.enc_mean_layer(h1)
        logvar = self.enc_logvar_layer(h1)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, c_dim, non_linear=False, init_logvar=-3):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dims = hidden_dim[::-1]
        self.non_linear = non_linear
        self.init_logvar = init_logvar
        self.c_dim = c_dim
        self.layer_sizes_decoder = self.hidden_dims + [input_dim]
        self.layer_sizes_decoder[0] = self.hidden_dims[0] + c_dim
        lin_layers = [
            nn.Linear(dim0, dim1, bias=True)
            for dim0, dim1 in zip(
                self.layer_sizes_decoder[:-1], self.layer_sizes_decoder[1:]
            )
        ]
        self.decoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.decoder_mean_layer = nn.Linear(
            self.layer_sizes_decoder[-2], self.layer_sizes_decoder[-1], bias=True
        )
        tmp_noise_par = torch.FloatTensor(1, self.input_size).fill_(self.init_logvar)
        self.logvar_out = Parameter(data=tmp_noise_par, requires_grad=True)

    def forward(self, z, c):
        c = c.reshape(-1, self.c_dim)
        x_rec = torch.cat((z, c), dim=1)
        for it_layer, layer in enumerate(self.decoder_layers):
            x_rec = layer(x_rec)
            if self.non_linear:
                x_rec = F.relu(x_rec)

        mu_out = self.decoder_mean_layer(x_rec)
        return Normal(loc=mu_out, scale=self.logvar_out.exp().pow(0.5))


class cVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        c_dim,
        learning_rate=0.001,
        non_linear=False,
    ):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            c_dim=c_dim,
            non_linear=non_linear,
        )
        self.decoder = Decoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            c_dim=c_dim,
            non_linear=non_linear,
        )
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )

    def encode(self, x, c):
        return self.encoder(x, c)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(z, c)

    def calc_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)

    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def forward(self, x, c):
        self.zero_grad()
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z, c)
        fwd_rtn = {"x_recon": x_recon, "mu": mu, "logvar": logvar}
        return fwd_rtn

    def sample_from_normal(self, normal):
        return normal.loc

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn["x_recon"]
        mu = fwd_rtn["mu"]
        logvar = fwd_rtn["logvar"]

        kl = self.calc_kl(mu, logvar)
        recon = self.calc_ll(x, x_recon)

        total = kl - recon
        losses = {"total": total, "kl": kl, "ll": recon}
        return losses

    def pred_latent(self, x, c, DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            mu, logvar = self.encode(x, c)
        latent = mu.cpu().detach().numpy()
        latent_var = logvar.exp().cpu().detach().numpy()
        return latent, latent_var

    def pred_recon(self, x, c, DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            mu, _ = self.encode(x, c)
            x_pred = self.decode(mu, c).loc.cpu().detach().numpy()
        return x_pred


class MyDataset_labels(Dataset):
    def __init__(self, data, labels, indices=False, transform=None):
        self.data = data
        self.labels = labels
        if isinstance(data, list) or isinstance(data, tuple):
            self.data = [
                torch.from_numpy(d).float() if isinstance(d, np.ndarray) else d
                for d in self.data
            ]
            self.N = len(self.data[0])
            self.shape = np.shape(self.data[0])
        elif isinstance(data, np.ndarray):
            self.data = torch.from_numpy(self.data).float()
            self.N = len(self.data)
            self.shape = np.shape(self.data)

        self.labels = torch.from_numpy(self.labels).long()

        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(self.data, list):
            x = [d[index] for d in self.data]
        else:
            x = self.data[index]

        if self.transform:
            x = self.transform(x)
        t = self.labels[index]
        if self.indices:
            return x, t, index
        return x, t

    def __len__(self):
        return self.N


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

    # if isinstance(model, nn.Module):
    #     first_layer_params = list(model.parameters())[0]
    #     print("Parameter values of the first layer:", first_layer_params)

    return model


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
    mean_recon = total_recon / len(val_loader)

    return mean_val_loss, mean_recon


def train(
    config,
    model,
    train_loader,
    val_loader,
    tolerance=50,
):

    model.to(DEVICE)

    best_val_loss = float("inf")

    best_mean_recon = float("-inf")

    epochs_no_improve = 0

    for epoch in range(1, config["epochs"] + 1):

        model.train()

        for batch_idx, batch in enumerate(train_loader):
            data_curr = batch[0].to(DEVICE)
            cov = batch[1].to(DEVICE)
            fwd_rtn = model.forward(data_curr, cov)
            loss = model.loss_function(data_curr, fwd_rtn)
            model.optimizer.zero_grad()
            loss["total"].backward()
            model.optimizer.step()

        val_loss, mean_recon = validate(model, val_loader, DEVICE)

        if val_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_loss

            best_mean_recon = mean_recon

            # print("New best val loss:", val_loss)
            # print("New best mean recon:", mean_recon)
            # print("at epoch:", epoch)

            # print("Improved at epoch:", epoch)

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= tolerance:
            # print("Early stopping at epoch:", epoch)

            break

    return best_val_loss, best_mean_recon


def train_k_fold(
    config,
    n_splits=10,
):

    data = pd.read_csv(
        Path(TRAIN_DATA_PATH),
        index_col=0,
        low_memory=False,
    )

    data["strata"] = (
        data["demo_sex_v2"].astype(str) + "_" + data["demo_comb_income_v2"].astype(str)
    )

    data["strata"] = pd.Categorical(data["strata"])

    train_strata = data.loc[TRAIN_SUBS, "strata"].cat.codes

    train_dataset = data.loc[
        TRAIN_SUBS,
        FEATURES,
    ].to_numpy()

    ### Covariate one hot encoding

    encoded_covariate_train = one_hot_encode_covariate(
        data,
        "demo_sex_v2",
        TRAIN_SUBS,
    )

    c_dim = encoded_covariate_train.shape[1]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold = 0
    total_loss = 0.0
    total_recon = 0.0

    ### TODO Stratified sampling
    for train_index, val_index in kf.split(train_dataset, train_strata):
        # print(f"Training on fold {fold+1}...")
        # Split dataset into training and validation sets for the current fold

        train_data, val_data = (
            train_dataset[train_index],
            train_dataset[val_index],
        )

        train_cov, val_cov = (
            encoded_covariate_train[train_index],
            encoded_covariate_train[val_index],
        )

        scaler = StandardScaler()

        train_data = scaler.fit_transform(train_data)

        val_data = scaler.transform(val_data)

        train_loader = DataLoader(
            MyDataset_labels(train_data, train_cov),
            batch_size=config["batch_size"],
            shuffle=True,
        )

        val_loader = DataLoader(
            MyDataset_labels(val_data, val_cov),
            batch_size=config["batch_size"],
            shuffle=False,
        )

        # Get input_dim based on the dataset
        input_dim = train_data.shape[1]

        model = build_model(
            config,
            input_dim,
            c_dim,
        )

        (
            val_loss,
            recon,
        ) = train(
            config,
            model,
            train_loader,
            val_loader,
        )

        print("val loss:", val_loss)

        total_loss += val_loss

        total_recon += recon

        fold += 1

    return total_loss / n_splits, total_recon / n_splits


def main(config):
    average_val_loss, average_recon = train_k_fold(config)

    return average_val_loss, average_recon


if __name__ == "__main__":

    project_title = args.project_title
    batch_sizes = args.batch_size
    learning_rates = args.learning_rate
    latent_dims = args.latent_dim
    hidden_dims = args.hidden_dim
    epochs = 5000

    results = []
    all_combinations = list(
        product(batch_sizes, learning_rates, latent_dims, hidden_dims)
    )

    # Loop through each combination with a progress bar
    for batch_size, learning_rate, latent_dim, hidden_dim in tqdm(
        all_combinations, desc="Processing combinations"
    ):
        config = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "latent_dim": latent_dim,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
        }
        # Assuming 'main' is your function that returns the validation loss and separation
        average_val_loss, average_recon = main(config)
        results.append((config, average_val_loss, average_recon))

    results_df = pd.DataFrame(
        results,
        columns=["config", "average_val_loss", "average_recon"],
    )

    results_df.to_csv(Path(processed_data_path.parent, f"{project_title}_results.csv"))

# TESTs
# 1. The combined modality dataframe works when selecting using features
# and subject ids.
# 2. The one hot encoding of the covariate is correct and correct in order.

# Local test command: python src/modelling/tune/ucl_cluster_tune/cVAE/cVAE_tune_local.py --data_path "data/processed_data" --feature_type "cortical_thickness" --project_title "local_grid_search_test"
# --batch_size "128" --learning_rate "0.005" --latent_dim "10" --hidden_dim "30"
