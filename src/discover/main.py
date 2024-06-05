from discover.scripts.test_deviation import discover

metrics = ["distance_deviation", "reconstruction_deviation", "uncertainty_deviation", ]

for metric in metrics:
    discover(metric=metric, if_low_entropy=False, dropout=False)
    discover(metric=metric, if_low_entropy=True, dropout=False)
    discover(metric=metric, if_low_entropy=False, dropout=True)
    discover(metric=metric, if_low_entropy=True, dropout=True)