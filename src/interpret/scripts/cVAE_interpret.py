# Feature: cortical_surface_area, Cohort: exter_test_subs, Significant Latent Dimensions: [4]

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from modelling.models.cVAE import cVAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def interpret_cVAE(
    dim_to_interpret: int,
    shift_how_much: float,
    checkpoint_path: Path,
    brain_features_of_interest: dict,
    feature_type: str,
    learning_rate: float,
    hidden_dim: list[int],
    latent_dim: int,
    c_dim: int = 2,
    num_std: int = 2,
    device=DEVICE,
) -> dict:

    feature_sets = {
        "t1w_cortical_thickness_rois": "cortical_thickness",
        "t1w_cortical_volume_rois": "cortical_volume",
        "t1w_cortical_surface_area_rois": "cortical_surface_area",
        "gordon_net_subcor_limbic_no_dup": "rsfmri",
    }

    features = brain_features_of_interest[feature_type]

    input_dim = len(features)

    model = cVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        c_dim=c_dim,
        learning_rate=learning_rate,
        non_linear=True,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path))

    mu = torch.zeros(1, latent_dim)
    mu[0, dim_to_interpret] = shift_how_much
    c = torch.zeros(1, 2)

    # 1 = Male Masculino; 2 = Female Femenino before one-hot encoding

    outlier_indices_by_sex = {"Male": [], "Female": []}

    for sex_code, sex in enumerate(["Male", "Female"]):

        feature_name = feature_sets[feature_type]

        c[0, sex_code] = 1

        recon = model.decode(mu, c).loc.cpu().detach().numpy()

        recon_flat = recon.flatten()

        mean = np.mean(recon_flat)
        std_dev = np.std(recon_flat)

        # Standardise the data
        recon_flat = (recon_flat - mean) / std_dev

        plt.figure(figsize=(10, 6))
        plt.hist(recon_flat, bins=30, alpha=0.75, color="blue", edgecolor="black")
        plt.title(
            "Frequency Distribution of Reconstructed Values After Standardisation"
        )
        plt.xlabel(f"{feature_name} Reconstructed Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        outlier_indices = np.where((recon_flat > num_std) | (recon_flat < -num_std))[0]

        outlier_indices_by_sex[sex] = outlier_indices.tolist()

    unique_outliers = set(outlier_indices_by_sex["Male"]).union(
        set(outlier_indices_by_sex["Female"])
    )

    outlier_indices_by_sex["Unique_indices"] = list(unique_outliers)

    return outlier_indices_by_sex


if __name__ == "__main__":

    processed_data_path = Path(
        "data",
        "processed_data",
    )

    brain_features_of_interest_path = Path(
        processed_data_path,
        "brain_features_of_interest.json",
    )

    checkpoint_path = Path(
        "checkpoints",
        "cortical_surface_area",
        "model_weights_no_dx.pt",
    )

    with open(brain_features_of_interest_path, "r") as brain_features_of_interest_file:
        brain_features_of_interest = json.load(brain_features_of_interest_file)

    feature_hypers = {
        "t1w_cortical_surface_area_rois": {
            "learning_rate": 0.0005,
            "latent_dim": 10,
            "hidden_dim": [30, 30],
        },
    }

    outlier_indices_by_sex = interpret_cVAE(
        dim_to_interpret=4,
        shift_how_much=-5,
        checkpoint_path=checkpoint_path,
        brain_features_of_interest=brain_features_of_interest,
        feature_type="t1w_cortical_surface_area_rois",
        learning_rate=feature_hypers["t1w_cortical_surface_area_rois"]["learning_rate"],
        latent_dim=feature_hypers["t1w_cortical_surface_area_rois"]["latent_dim"],
        hidden_dim=feature_hypers["t1w_cortical_surface_area_rois"]["hidden_dim"],
        num_std=1,
    )

    with open(
        Path("src", "interpret", "results", "dim_4_surface_area_indices_by_sex.json"),
        "w",
    ) as results_file:
        json.dump(outlier_indices_by_sex, results_file)
