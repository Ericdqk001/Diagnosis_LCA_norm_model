import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from nichord.chord import plot_chord

# Load brain features of interest
brain_features_of_interest_path = Path(
    "data",
    "processed_data",
    "brain_features_of_interest.json",
)

with open(brain_features_of_interest_path) as f:
    brain_features_of_interest = json.loads(f.read())

fc_features = brain_features_of_interest["gordon_net_subcor_limbic_no_dup"]

low_entropy = False

if low_entropy:

    # Load feature effect sizes
    fc_feature_effect_sizes_path = Path(
        "src",
        "discover",
        "results",
        "low_entropy",
        "sig_ind_regions",
        "rsfmri_significant_regions.csv",
    )

else:
    fc_feature_effect_sizes_path = Path(
        "src",
        "discover",
        "results",
        "sig_ind_regions",
        "rsfmri_significant_regions.csv",
    )

fc_feature_effect_sizes_df = pd.read_csv(fc_feature_effect_sizes_path)

cohorts = ["inter_test", "exter_test", "high_test"]

for cohort in cohorts:

    cohort_fc_feature_effect_sizes_df = fc_feature_effect_sizes_df[
        fc_feature_effect_sizes_df["Group"] == cohort
    ]

    # Create an empty list of length 143 with all values set to 0
    weights = [0] * 143

    # Extract the numerical indices from the "Metric" column
    sig_feature_indices = (
        cohort_fc_feature_effect_sizes_df["Metric"]
        .str.extract(r"(\d+)$")
        .astype(int)[0]
    )

    # Iterate through the extracted indices and the corresponding effect sizes
    for idx, effect_size in zip(
        sig_feature_indices, cohort_fc_feature_effect_sizes_df["Mean Effect Size"]
    ):
        weights[idx] = effect_size

    # Dictionary mapping indices to labels
    idx_to_label = {
        0: "AUD",  # auditory network
        1: "CON",  # cingulo-opercular network
        2: "CPN",  # cingulo-parietal network
        3: "DMN",  # default mode network
        4: "DAN",  # dorsal attention network
        5: "FPN",  # fronto-parietal network
        6: "NN",  # none network
        7: "RTN",  # retrosplenial-temporal network
        8: "SN",  # salience network
        9: "SHN",  # sensorimotor hand network
        10: "SMN",  # sensorimotor mouth network
        11: "VAN",  # ventral attention network
        12: "VIS",  # visual network
        13: "LAA",  # Left accumbens area
        14: "RAA",  # Right accumbens area
        15: "LAC",  # Left amygdala
        16: "RAC",  # Right amygdala
    }

    # Edges without duplicates
    edges = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 9),
        (0, 10),
        (0, 11),
        (0, 12),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 10),
        (1, 11),
        (1, 12),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 10),
        (2, 11),
        (2, 12),
        (3, 3),
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 7),
        (3, 8),
        (3, 9),
        (3, 10),
        (3, 11),
        (3, 12),
        (4, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8),
        (4, 9),
        (4, 10),
        (4, 11),
        (4, 12),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 8),
        (5, 9),
        (5, 10),
        (5, 11),
        (5, 12),
        (6, 6),
        (6, 7),
        (6, 8),
        (6, 9),
        (6, 10),
        (6, 11),
        (6, 12),
        (7, 7),
        (7, 8),
        (7, 9),
        (7, 10),
        (7, 11),
        (7, 12),
        (8, 8),
        (8, 9),
        (8, 10),
        (8, 11),
        (8, 12),
        (9, 9),
        (9, 10),
        (9, 11),
        (9, 12),
        (10, 10),
        (10, 11),
        (10, 12),
        (11, 11),
        (11, 12),
        (12, 12),
        (0, 13),
        (0, 14),
        (0, 15),
        (0, 16),
        (1, 13),
        (1, 14),
        (1, 15),
        (1, 16),
        (2, 13),
        (2, 14),
        (2, 15),
        (2, 16),
        (3, 13),
        (3, 14),
        (3, 15),
        (3, 16),
        (4, 13),
        (4, 14),
        (4, 15),
        (4, 16),
        (5, 13),
        (5, 14),
        (5, 15),
        (5, 16),
        (6, 13),
        (6, 14),
        (6, 15),
        (6, 16),
        (7, 13),
        (7, 14),
        (7, 15),
        (7, 16),
        (8, 13),
        (8, 14),
        (8, 15),
        (8, 16),
        (9, 13),
        (9, 14),
        (9, 15),
        (9, 16),
        (10, 13),
        (10, 14),
        (10, 15),
        (10, 16),
        (11, 13),
        (11, 14),
        (11, 15),
        (11, 16),
        (12, 13),
        (12, 14),
        (12, 15),
        (12, 16),
    ]

    # Filter out edges with a weight of 0
    filtered_edges = [
        (edge, weight) for edge, weight in zip(edges, weights) if weight != 0
    ]
    filtered_edges, filtered_weights = zip(*filtered_edges)

    # Normalize weights for linewidths
    min_weight = min(filtered_weights)
    max_weight = max(filtered_weights)
    normalized_weights = [
        (weight - min_weight) / (max_weight - min_weight) * 7
        for weight in filtered_weights
    ]

    # Plot the chord diagram
    fp_chord = "ex0_chord.png"
    plot_chord(
        idx_to_label,
        filtered_edges,
        edge_weights=filtered_weights,
        fp_chord=None,
        linewidths=normalized_weights,
        alphas=0.9,
        do_ROI_circles=True,
        label_fontsize=70,
        do_ROI_circles_specific=True,
        ROI_circle_radius=0.02,
    )
    plt.show()
