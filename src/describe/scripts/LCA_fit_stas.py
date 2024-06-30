from pathlib import Path

import pandas as pd

fit_stats_path = Path(
    "data",
    "LCA",
    "LCA_model_fit_statistics_LMR.csv",
)

# Read the CSV file
df = pd.read_csv(fit_stats_path)

# Round numerical values for consistency
df = df.round(2)

# Convert 'NA' in 'LMR_PValue' to empty strings for clarity in the table
df["LMR_PValue"] = df["LMR_PValue"].replace("NA", "")

# Drop columns

df = df.drop(
    columns=[
        "Npar",
        "AWE",
    ]
)

# Generate the LaTeX table
latex_table = df.to_latex(
    index=False,
    column_format="lccccccccc",
    header=True,
    caption="Fit statistics for latent class models with 2 to 6 classes.",
    label="tab:fit_statistics",
    float_format="%.2f",
)

print(latex_table)


### Classification diagnostics

cls_diag_path = Path(
    "data",
    "LCA",
    "LCA_model_classification_diagnostics.csv",
)

# Read the CSV file
df = pd.read_csv(cls_diag_path)

