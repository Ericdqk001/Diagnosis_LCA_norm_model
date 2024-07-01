from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

lca_path = Path(
    "data",
    "LCA",
)

lca_class_memberships_path = Path(
    lca_path,
    "cbcl_class_member_prob.csv",
)

lca_class_memberships = pd.read_csv(
    lca_class_memberships_path,
    index_col=0,
    low_memory=False,
)

data_path = Path(
    "data",
    "raw_data",
    "core",
    "mental-health",
    "mh_p_cbcl.csv",
)

cbcl_t_vars_path = Path(
    "data",
    "var_dict",
    "cbcl_8_dim_t.csv",
)

cbcl = pd.read_csv(
    data_path,
    index_col=0,
    low_memory=False,
)

cbcl_t_vars_df = pd.read_csv(cbcl_t_vars_path)

cbcl_t_vars = cbcl_t_vars_df["var_name"].tolist()

# Add the internalising and externalising syndromes and total problem scales to the
## list for analysis
sum_syndrome = [
    "cbcl_scr_syn_internal_t",
    "cbcl_scr_syn_external_t",
    "cbcl_scr_syn_totprob_t",
]

cbcl_t_vars.extend(sum_syndrome)

# Select the baseline data
baseline_cbcl = cbcl[cbcl["eventname"] == "baseline_year_1_arm_1"]

# Filter columns with t variables
filtered_cbcl = baseline_cbcl[cbcl_t_vars]

# Drop the dummy variables from lca first with the same cbcl variable names

lca_class_memberships = lca_class_memberships.drop(
    cbcl_t_vars,
    axis=1,
)

# Join the LCA class memberships with the CBCL data

cbcl_lca_memberships = lca_class_memberships.join(
    filtered_cbcl,
    how="inner",
)


# Plot the t-scores for the CBCL variables for each LCA class


def plot_cbcl_means(
    df,
    cbcl_scales=cbcl_t_vars,
    low_entropy=False,
):

    if low_entropy == True:
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

        df = df[~df.index.isin(high_entropy_subs)]

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

    bright_colors = [
        "#FF5733",
        "#33FF57",
        "#3357FF",
        "#FF33A1",
        "#FFDB33",
        "#33FFF0",
        "#FF3333",
        "#8D33FF",
    ]

    class_names = [
        "Low Symptom",
        "Predominantly Internalising",
        "Predominantly Externalising",
        "Highly Dysregulated",
    ]

    plt.figure(figsize=(10, 6.7))

    for class_id in sorted(df["predicted_class"].unique()):
        class_df = df[df["predicted_class"] == class_id]
        means = class_df[cbcl_scales].mean()
        plt.plot(
            cbcl_scales,
            means,
            marker="o",
            markersize=10,
            markerfacecolor="none",  # Make the marker an empty circle
            markeredgewidth=2,
            markeredgecolor=bright_colors[
                class_id % len(bright_colors)
            ],  # Color the marker edge
            color=bright_colors[class_id % len(bright_colors)],
            label=class_names[class_id - 1],
            linewidth=4,  # Increase the line width
        )

    plt.title("CBCL Scale Mean Scores by Class", color="black", fontweight="bold")
    plt.xlabel("CBCL Scales", color="black", fontweight="bold")
    plt.ylabel("Mean Scores", color="black", fontweight="bold")
    plt.xticks(rotation=45, color="black", fontweight="bold")
    plt.yticks(color="black", fontweight="bold")
    plt.legend(prop={"weight": "bold"})  # Set legend labels to bold
    plt.grid(False)  # This line removes the grid
    plt.gca().set_facecolor("white")  # This sets the plot background color to white
    plt.tight_layout()
    plt.show()


# Example usage, assuming lca_psych_dx_low_entropy is the DataFrame you've prepared earlier
plot_cbcl_means(cbcl_lca_memberships, low_entropy=False)
