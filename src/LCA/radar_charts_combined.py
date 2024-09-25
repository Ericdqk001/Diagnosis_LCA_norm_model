from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# Data Preparation Function
def prepare_combined_data():
    # Load the datasets
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

    # Filter columns with T variables
    filtered_cbcl = baseline_cbcl[cbcl_t_vars]

    # Drop dummy variables from LCA that have the same CBCL variable names
    lca_class_memberships = lca_class_memberships.drop(cbcl_t_vars, axis=1)

    # Merge LCA class memberships with CBCL data
    cbcl_lca_memberships = lca_class_memberships.join(filtered_cbcl, how="inner")

    # Join the LCA class memberships with the psychiatric diagnoses data
    lca_psych_dx_combined = cbcl_lca_memberships.join(psych_dx, how="inner")

    return lca_psych_dx_combined


# Combined Radar Plot Function
def plot_combined_radar_charts(df, threshold=65, low_entropy=False):
    # Define the scales and diagnoses
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

    diagnoses = [
        "Has_ADHD",
        "Has_Depression",
        "Has_Bipolar",
        "Has_Anxiety",
        "Has_OCD",
        "Has_ASD",
        "Has_DBD",
    ]

    short_diagnoses = [diag.replace("Has_", "") for diag in diagnoses]

    class_names = [
        "Predominantly Internalizing",
        "Predominantly Externalizing",
        "Highly Dysregulated",
    ]

    # Use specified colors and line styles
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    line_styles = ["-", "--", ":"]

    # Filter out high entropy subjects if required
    if low_entropy:
        high_entropy_subs_path = Path("data", "LCA", "subjects_with_high_entropy.csv")
        high_entropy_subs = pd.read_csv(high_entropy_subs_path, low_memory=False)[
            "subject"
        ].tolist()
        df = df[~df.index.isin(high_entropy_subs)]

    num_vars_cbcl = len(cbcl_scales)
    angles_cbcl = np.linspace(0, 2 * np.pi, num_vars_cbcl, endpoint=False).tolist()
    angles_cbcl += angles_cbcl[:1]  # Ensure the plot is closed

    num_vars_diag = len(diagnoses)
    angles_diag = np.linspace(0, 2 * np.pi, num_vars_diag, endpoint=False).tolist()
    angles_diag += angles_diag[:1]  # Ensure the plot is closed

    fig, axes = plt.subplots(
        1, 2, subplot_kw=dict(polar=True), figsize=(18, 8), dpi=100
    )

    class_ids_to_plot = [2, 3, 4]

    # Plot CBCL scales radar chart (merged cohorts)
    ax = axes[0]
    for i, class_id in enumerate(class_ids_to_plot):
        class_df = df[df["predicted_class"] == class_id]

        proportions_cbcl = (class_df[cbcl_scales] >= threshold).mean().tolist()
        proportions_cbcl += proportions_cbcl[:1]

        ax.plot(
            angles_cbcl,
            proportions_cbcl,
            marker="o",
            linestyle=line_styles[i],
            color=colors[i],
            alpha=0.8,
            linewidth=2,
        )
        ax.fill(angles_cbcl, proportions_cbcl, color=colors[i], alpha=0.3)

    # (a) At-risk CBCL syndrome proportions

    ax.set_title("", fontweight="bold", fontsize=16)
    ax.set_xticks(angles_cbcl[:-1])
    ax.set_xticklabels(axis_labels, rotation=45, fontsize=12)

    ax.set_ylim([0, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="black", size=10)

    ax.grid(True)

    # Plot Diagnosis proportions radar chart (merged cohorts)
    ax = axes[1]
    for i, class_id in enumerate(class_ids_to_plot):
        class_df = df[df["predicted_class"] == class_id]
        proportions_diag = class_df[diagnoses].mean().tolist()
        proportions_diag += proportions_diag[:1]

        ax.plot(
            angles_diag,
            proportions_diag,
            marker="o",
            linestyle=line_styles[i],
            color=colors[i],
            alpha=0.8,
            linewidth=2,
        )
        ax.fill(angles_diag, proportions_diag, color=colors[i], alpha=0.3)
    # (b) Diagnosis proportions
    ax.set_title("", fontweight="bold", fontsize=16)
    ax.set_xticks(angles_diag[:-1])
    ax.set_xticklabels(short_diagnoses, rotation=45, fontsize=12)

    ax.set_ylim([0, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="black", size=10)

    ax.grid(True)

    # Create custom legend handles with the correct colors and line styles
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[i],
            linestyle=line_styles[i],
            linewidth=2,
            marker="o",
            label=class_names[i],
        )
        for i in range(len(class_names))
    ]

    # Consolidated legend for both charts
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=12,
        frameon=False,
    )

    fig.suptitle(
        "",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjusted to make room for the legend
    plt.show()


# Example Usage
df_combined = prepare_combined_data()
plot_combined_radar_charts(df_combined, threshold=65, low_entropy=True)
