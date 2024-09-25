from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Load the data
psych_dx_path = Path("data", "liza_data", "all_psych_dx_r5.csv")
psych_dx = pd.read_csv(psych_dx_path, index_col=0, low_memory=False)

lca_path = Path("data", "LCA")
lca_class_memberships_path = Path(lca_path, "cbcl_class_member_prob.csv")
lca_class_memberships = pd.read_csv(
    lca_class_memberships_path, index_col=0, low_memory=False
)

data_path = Path("data", "raw_data", "core", "mental-health", "mh_p_cbcl.csv")
cbcl_t_vars_path = Path("data", "var_dict", "cbcl_8_dim_t.csv")
cbcl = pd.read_csv(data_path, index_col=0, low_memory=False)
cbcl_t_vars_df = pd.read_csv(cbcl_t_vars_path)
cbcl_t_vars = cbcl_t_vars_df["var_name"].tolist()

# Select the baseline data
baseline_cbcl = cbcl[cbcl["eventname"] == "baseline_year_1_arm_1"]

# Filter columns with t variables
filtered_cbcl = baseline_cbcl[cbcl_t_vars]

# Drop the dummy variables from lca first with the same cbcl variable names
lca_class_memberships = lca_class_memberships.drop(cbcl_t_vars, axis=1)

# Join the LCA class memberships with the CBCL data
cbcl_lca_memberships = lca_class_memberships.join(filtered_cbcl, how="inner")


def plot_individual_cbcl_patterns_combined(df, low_entropy=False):
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

    syndrome_map = {
        "cbcl_scr_syn_anxdep_t": "Anxiety/Depress",
        "cbcl_scr_syn_withdep_t": "Withdraw/Depress",
        "cbcl_scr_syn_somatic_t": "Somatic",
        "cbcl_scr_syn_social_t": "Social",
        "cbcl_scr_syn_thought_t": "Thought",
        "cbcl_scr_syn_attention_t": "Attention",
        "cbcl_scr_syn_rulebreak_t": "RuleBreak",
        "cbcl_scr_syn_aggressive_t": "Aggressive",
    }

    axis_labels = [syndrome_map[scale] for scale in cbcl_scales]

    # Load subjects with high entropy
    high_entropy_subs_path = Path("data", "LCA", "subjects_with_high_entropy.csv")
    high_entropy_subs = pd.read_csv(high_entropy_subs_path, low_memory=False)[
        "subject"
    ].tolist()

    # Separate the dataframe into high entropy and non-high entropy subjects
    high_entropy_df = df[df.index.isin(high_entropy_subs)]
    low_entropy_df = df[~df.index.isin(high_entropy_subs)]

    class_names = [
        "Class 2",
        "Class 3",
        "Class 4",
    ]

    bright_colors = [
        "green",  # Class 2
        "red",  # Class 3
        "purple",  # Class 4
    ]

    y_min = df[cbcl_scales].min().min()
    y_max = df[cbcl_scales].max().max()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes = axes.flatten()

    for i, class_id in enumerate(sorted(df["predicted_class"].unique())):
        if class_id == 1:
            continue  # Skip Class 1

        ax = axes[i - 1]

        # Plot the background with high entropy subjects in grey
        class_df_bg = high_entropy_df[high_entropy_df["predicted_class"] == class_id]
        for _, row in class_df_bg.iterrows():
            ax.plot(
                axis_labels,
                row[cbcl_scales],
                color="grey",
                alpha=0.1,
                linewidth=1,
            )

        # Overlay the low entropy subjects in bright colors
        class_df_fg = low_entropy_df[low_entropy_df["predicted_class"] == class_id]
        for _, row in class_df_fg.iterrows():
            ax.plot(
                axis_labels,
                row[cbcl_scales],
                color=bright_colors[class_id - 2],
                alpha=0.8,
                linewidth=1,
            )

        ax.set_title(f"{class_names[class_id - 2]}", fontweight="bold")
        ax.set_xticks(range(len(cbcl_scales)))
        ax.set_xticklabels(axis_labels, rotation=45)
        ax.set_ylabel("Scores")
        ax.set_ylim(y_min, y_max)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# Plot the combined figure
plot_individual_cbcl_patterns_combined(cbcl_lca_memberships)
