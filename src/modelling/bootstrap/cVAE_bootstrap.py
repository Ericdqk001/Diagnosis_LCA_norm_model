import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.distributions import Normal
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset
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


### MODEL AND CUSTOM DATASET


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


class DropoutDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        c_dim,
        non_linear=False,
        init_logvar=-3,
        dropout=False,
        dropout_rate=0.20,
    ):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dims = hidden_dim[::-1]
        self.non_linear = non_linear
        self.dropout = dropout
        self.init_logvar = init_logvar
        self.c_dim = c_dim
        self.dropout_rate = dropout_rate
        self.layer_sizes_decoder = self.hidden_dims + [input_dim]
        self.layer_sizes_decoder[0] = self.hidden_dims[0] + c_dim

        # Create layers for the decoder
        layers = []
        for dim0, dim1 in zip(
            self.layer_sizes_decoder[:-1], self.layer_sizes_decoder[1:-1]
        ):
            layers.append(nn.Linear(dim0, dim1, bias=True))
            if self.non_linear:
                layers.append(nn.ReLU())
            if self.dropout:
                layers.append(nn.Dropout(self.dropout_rate))

        # Adding the final linear layer separately
        self.decoder_layers = nn.Sequential(*layers)
        self.decoder_mean_layer = nn.Linear(
            self.layer_sizes_decoder[-2], self.layer_sizes_decoder[-1], bias=True
        )

        # Parameter for the output noise
        tmp_noise_par = torch.FloatTensor(1, self.input_size).fill_(self.init_logvar)
        self.logvar_out = Parameter(data=tmp_noise_par, requires_grad=True)

    def forward(self, z, c):
        c = c.reshape(-1, self.c_dim)
        x_rec = torch.cat((z, c), dim=1)

        # Pass through decoder layers
        x_rec = self.decoder_layers(x_rec)

        # Output mean from the last layer
        mu_out = self.decoder_mean_layer(x_rec)

        # Create a normal distribution with the mean and standard deviation
        return Normal(loc=mu_out, scale=self.logvar_out.exp().pow(0.5))


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
        dropout=False,
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

        if dropout:
            self.decoder = DropoutDecoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                c_dim=c_dim,
                non_linear=non_linear,
                dropout=True,
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
            # x_std = self.decode(mu, c).scale.cpu().detach().numpy()
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


###


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


def reconstruction_deviation(x, x_pred):

    dev = np.mean((x - x_pred) ** 2, axis=1)

    return dev


def ind_reconstruction_deviation(x, x_pred):
    dev = (x - x_pred) ** 2

    return dev


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
        ]

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
