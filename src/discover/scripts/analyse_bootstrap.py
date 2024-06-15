from pathlib import Path

import pandas as pd

bootstrap_result_path = Path(
    "src",
    "discover",
    "results",
    "bootstrap",
    "cortical_thickness_bootstrap_results.csv",
)

bootstrap_result = pd.read_csv(
    bootstrap_result_path,
    index_col=0,
)
