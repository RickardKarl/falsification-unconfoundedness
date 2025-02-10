import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def load_data(data_folder) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # processed data by Louizos et al (2017)

    df_covar = pd.read_csv(
        os.path.join(data_folder, "twin_pairs_X_3years_samesex.csv")
    ).drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)

    # drop id of infants in covariates
    df_covar.drop(["infant_id_0", "infant_id_1", "data_year"], axis=1, inplace=True)

    # Impute missing valuess
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputed = imputer.fit_transform(df_covar)
    df_covar = pd.DataFrame(imputed, columns=df_covar.columns)
    df_covar = (df_covar - df_covar.mean()) / df_covar.std()  # centralize

    return df_covar


def get_env_sample_indices(df, env_label: str) -> dict:
    environment_labels = df[env_label].unique()
    env_idx = {}
    threshold_nbr_samples = 100

    for e in range(len(environment_labels)):
        sample_idx = df[df[env_label] == environment_labels[e]].index.values
        if len(sample_idx) >= threshold_nbr_samples:
            env_idx[e] = sample_idx

    return env_idx
