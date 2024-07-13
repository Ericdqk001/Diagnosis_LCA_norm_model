from pathlib import Path

import pandas as pd
import statsmodels.api as sm

cVAE_discover_results_path = Path(
    "src",
    "discover",
    "results",
)

feature_sets = {
    "t1w_cortical_thickness_rois": "Cortical Thickness",
    "t1w_cortical_volume_rois": "Cortical Volume",
    "t1w_cortical_surface_area_rois": "Cortical Surface Area",
    "gordon_net_subcor_limbic_no_dup": "Functional Connectivity",
}

results_path = Path(
    cVAE_discover_results_path,
    "low_entropy",
)

output_data_save_path = Path(
    cVAE_discover_results_path,
    "output_data_low_entropy",
)

# Initialize a DataFrame to store the p-values and coefficients
results_df = pd.DataFrame(
    columns=[
        "Feature",
        "Latent Deviation p-value",
        "Latent Deviation Coefficient",
        "Reconstruction Deviation p-value",
        "Reconstruction Deviation Coefficient",
    ]
)

for feature in feature_sets:

    print(f"Discovering feature: {feature}")

    feature_output_data_save_path = Path(
        output_data_save_path,
        f"{feature}_output_data_with_dev.csv",
    )

    output_data = pd.read_csv(feature_output_data_save_path)

    biological_sex = output_data["sex"].to_numpy() - 1

    latent_deviation = output_data["mahalanobis_distance"].to_numpy()
    recon_deviation = output_data["reconstruction_deviation"].to_numpy()

    # Fit logistic regression models
    X_latent = sm.add_constant(latent_deviation, has_constant="add")
    X_latent = pd.DataFrame(X_latent, columns=["const", "mahalanobis_distance"])
    X_recon = sm.add_constant(recon_deviation, has_constant="add")
    X_recon = pd.DataFrame(X_recon, columns=["const", "reconstruction_deviation"])

    model_latent = sm.Logit(biological_sex, X_latent).fit(disp=0)
    model_recon = sm.Logit(biological_sex, X_recon).fit(disp=0)

    p_value_latent = model_latent.pvalues["mahalanobis_distance"]
    coeff_latent = model_latent.params["mahalanobis_distance"]
    p_value_recon = model_recon.pvalues["reconstruction_deviation"]
    coeff_recon = model_recon.params["reconstruction_deviation"]

    # Create a DataFrame with the new p-values and coefficients
    new_row = pd.DataFrame(
        {
            "Feature": [feature_sets[feature]],
            "Latent Deviation p-value": [p_value_latent],
            "Latent Deviation Coefficient": [coeff_latent],
            "Reconstruction Deviation p-value": [p_value_recon],
            "Reconstruction Deviation Coefficient": [coeff_recon],
        }
    )

    # Concatenate the new row to the results_df
    results_df = pd.concat([results_df, new_row], ignore_index=True)

print(results_df)

test_results_save_path = Path(
    "src",
    "test",
    "results",
)

if not test_results_save_path.exists():
    test_results_save_path.mkdir(parents=True)

# Save the results to a CSV file
results_save_path = Path(
    test_results_save_path, "p_values_coefficients_deconfounding_assessment.csv"
)
results_df.to_csv(results_save_path, index=False)
