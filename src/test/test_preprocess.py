### Test if de-confounding worked
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import StratifiedKFold
from statsmodels.formula.api import ols

processed_data_path = Path(
    "data",
    "processed_data",
)

brain_features_of_interest_path = Path(
    processed_data_path,
    "brain_features_of_interest.json",
)

with open(brain_features_of_interest_path, "r") as f:
    brain_features_of_interest = json.load(f)


# %%
# Load de-confounded data
t1w_cortical_features_resid_path = Path(
    processed_data_path,
    "t1w_cortical_features_resid_exc_sex.csv",
)

t1w_cortical_features_resid = pd.read_csv(
    t1w_cortical_features_resid_path,
    index_col=0,
    low_memory=False,
)

gordon_cor_subcortical_resid_path = Path(
    "data",
    "processed_data",
    "gordon_cor_subcortical_resid_exc_sex.csv",
)

gordon_cor_subcortical_resid = pd.read_csv(
    gordon_cor_subcortical_resid_path,
    index_col=0,
    low_memory=False,
)

t1w_cortical_thickness_rois = brain_features_of_interest["t1w_cortical_thickness_rois"]
t1w_cortical_volume_rois = brain_features_of_interest["t1w_cortical_volume_rois"]
t1w_cortical_surface_area_rois = brain_features_of_interest[
    "t1w_cortical_surface_area_rois"
]
gordon_net_subcor_no_dup = brain_features_of_interest["gordon_net_subcor_no_dup"]

# %%
# De-confounded data
# Test if brain features can predict site

# Cortical thickness
X = np.asarray(t1w_cortical_features_resid[t1w_cortical_thickness_rois])
y = np.asarray(t1w_cortical_features_resid["label_site"])
skf = StratifiedKFold(n_splits=5)
gen_split = skf.split(X, y)
list_mcc = []
for i, (train_index, test_index) in enumerate(gen_split):
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]
    RF = RFC()
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    print(mcc(y_test, y_pred))
    list_mcc.append(mcc(y_test, y_pred))
print("Cortical Thickness Mean MCC is ", np.mean(np.asarray(list_mcc)))

# Results
# Original data:
# De-confounded data: Mean MCC is  0.0026443056227516567

# Cortical volume


X = np.asarray(t1w_cortical_features_resid[t1w_cortical_volume_rois])
y = np.asarray(t1w_cortical_features_resid["label_site"])
skf = StratifiedKFold(n_splits=5)
gen_split = skf.split(X, y)
list_mcc = []
for i, (train_index, test_index) in enumerate(gen_split):
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]
    RF = RFC()
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    print(mcc(y_test, y_pred))
    list_mcc.append(mcc(y_test, y_pred))
print("Cortical Volume Mean MCC is ", np.mean(np.asarray(list_mcc)))

# rsfmri data

X = np.asarray(gordon_cor_subcortical_resid[gordon_net_subcor_no_dup])
y = np.asarray(gordon_cor_subcortical_resid["label_site"])
skf = StratifiedKFold(n_splits=5)
gen_split = skf.split(X, y)
list_mcc = []
for i, (train_index, test_index) in enumerate(gen_split):
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]
    RF = RFC()
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    print(mcc(y_test, y_pred))
    list_mcc.append(mcc(y_test, y_pred))
print("Rsfmri Mean MCC is ", np.mean(np.asarray(list_mcc)))

# Results
# Mean MCC is  0.004345320006670439
# %%
# Test the effect of regressing out other confounders

# Cortical thickness
# t1w_cortical_thickness_resid

t1w_r2 = []

for img_feature in t1w_cortical_volume_rois:

    formula = (
        "%s ~ C(demo_sex_v2) + smri_vol_scs_intracranialv + interview_age" % img_feature
    )
    model = ols(formula, t1w_cortical_features_resid).fit()

    t1w_r2.append(model.rsquared)

print("Mean R2 is ", np.mean(np.asarray(t1w_r2)))

# Results
# Mean R2 is  -3.293161539259755e-16

# Cortical volume

t1w_cv_r2 = []

for img_feature in t1w_cortical_thickness_rois:

    formula = (
        "%s ~ C(demo_sex_v2) + smri_vol_scs_intracranialv + interview_age" % img_feature
    )
    model = ols(formula, t1w_cortical_features_resid).fit()

    t1w_cv_r2.append(model.rsquared)

print("Mean R2 is ", np.mean(np.asarray(t1w_cv_r2)))


# Surface area

t1w_sa_r2 = []

for img_feature in t1w_cortical_surface_area_rois:

    formula = (
        "%s ~ C(demo_sex_v2) + smri_vol_scs_intracranialv + interview_age" % img_feature
    )
    model = ols(formula, t1w_cortical_features_resid).fit()

    t1w_sa_r2.append(model.rsquared)

print("Mean R2 is ", np.mean(np.asarray(t1w_sa_r2)))


# rsfmri data

rsfmri_r2 = []

for img_feature in gordon_net_subcor_no_dup:

    formula = (
        "%s ~ C(demo_sex_v2) + smri_vol_scs_intracranialv + interview_age" % img_feature
    )
    model = ols(formula, gordon_cor_subcortical_resid).fit()

    rsfmri_r2.append(model.rsquared)

print("Mean R2 is ", np.mean(np.asarray(rsfmri_r2)))

# Results
# Mean R2 is  4.00731387586595e-17


# Test the multicollinearity between age and intracranial volume

age = t1w_cortical_features_resid["interview_age"]

intracranialv = t1w_cortical_features_resid["smri_vol_scs_intracranialv"]

age_intracranialv_corr = np.corrcoef(age, intracranialv)

print("Correlation between age and intracranial volume is ", age_intracranialv_corr)
