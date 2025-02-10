import numpy as np
import pandas as pd


from experiments.data.abstract import AbstractData
from experiments.data.twins.utils import load_data, get_env_sample_indices
from src.ours.utils import create_polynomial_representation


class DGP(AbstractData):

    COVAR_LIST = [
        "birmon",
        "dfageq",
        "dlivord_min",
        "dtotord_min",
        "gestat10",
        "mager8",
        "meduc6",
        "mplbir",
        "nprevistq",
        # "stoccfipb",  # bad
    ]
    ENV_LABEL_TWINS = "brstate"

    def __init__(
        self,
        degree=1,
        conf_strength: float = 1.0,
        transportability_violation: float = 0.0,
        n_envs: int = 25,
        n_observed_confounders: int = 5,
        use_twins_covariates: bool = False,
        data_folder: str = None,
        seed: int = None,
    ) -> None:
        self.degree = degree
        self.transportability_violation = transportability_violation
        self.conf_strength = conf_strength
        self.n_envs = n_envs
        self.n_observed_confounders = n_observed_confounders
        self.use_twins_covariates = use_twins_covariates
        self.random_state = np.random.RandomState(seed)

        if self.use_twins_covariates is False:
            self.covar_list = [f"X_{i}" for i in range(self.n_observed_confounders)]
        else:
            assert data_folder is not None, f"data_folder must provided"
            self.df_covar = load_data(data_folder=data_folder)
            self.env_idx = get_env_sample_indices(self.df_covar, DGP.ENV_LABEL_TWINS)
            self.n_envs = len(self.env_idx.keys())  # override number of environments
            self.covar_list = None  # Will be set after sampling

    def get_covar(self) -> list:
        assert self.covar_list is not None
        return self.covar_list

    def random_select_Twins_confounders(self) -> list:
        assert (
            self.n_observed_confounders <= 5
        ), "Maximum number of unobserved confounder allowed to match with setup of Karlsson and Krijthe (2023)"
        covar_list = list(
            map(
                str,
                self.random_state.choice(DGP.COVAR_LIST, 5, replace=False),
            )
        )

        measured_covar = list(
            map(
                str,
                self.random_state.choice(
                    covar_list, self.n_observed_confounders, replace=False
                ),
            )
        )
        unmeasured_covar = [c for c in covar_list if c not in measured_covar]

        return measured_covar, unmeasured_covar

    def sample(
        self,
        n_samples: int,
        sigma_coefs={
            "sigma_X_to_A": 0.0,
            "sigma_X_to_Y": 0.0,
            "sigma_A_to_Y": 0.0,
            "sigma_intercept_A": 1,
            "sigma_intercept_Y": 1,
        },
    ) -> dict:

        ##############################################
        # Sample parameters for environments
        ##############################################

        # Coefs from X to A and Y
        if self.degree == 1:
            x_transform = lambda x: x
        else:
            x_transform = lambda x: create_polynomial_representation(x, self.degree)

        test_vector = self.random_state.multivariate_normal(
            np.zeros((self.n_observed_confounders)),
            np.eye(self.n_observed_confounders),
            size=(1),
        )
        feature_representation_dim = x_transform(test_vector).shape[1]

        # randomize coefs. for X
        x_to_a_coef = self.random_state.choice(
            [-1, 1], size=(feature_representation_dim, 1)
        )
        x_to_y_coef = np.ones(shape=(feature_representation_dim, 1))
        a_to_y_effect = np.ones(shape=(1, 1))

        #
        x_to_a_coef = x_to_a_coef + self.random_state.normal(
            0,
            sigma_coefs["sigma_X_to_A"],
            size=(feature_representation_dim, self.n_envs),
        )
        x_to_y_coef = x_to_y_coef + self.random_state.normal(
            0,
            sigma_coefs["sigma_X_to_Y"],
            size=(feature_representation_dim, self.n_envs),
        )
        a_to_y_effect = a_to_y_effect + self.random_state.normal(
            0, sigma_coefs["sigma_A_to_Y"], size=(1, self.n_envs)
        )

        # Set random intercepts
        intercept_a = self.random_state.normal(
            0, sigma_coefs["sigma_intercept_A"], size=(1, self.n_envs)
        )
        intercept_y = self.random_state.normal(
            0, sigma_coefs["sigma_intercept_A"], size=(1, self.n_envs)
        )

        # If not use Twins
        if self.use_twins_covariates is False:
            mu_X = self.random_state.normal(
                0, 1.0, size=(self.n_envs, self.n_observed_confounders)
            )
            mu_U = self.random_state.normal(0, 1.0, size=(self.n_envs, 1))
            X_cov = np.full(
                (self.n_observed_confounders, self.n_observed_confounders), 0.1
            )
            np.fill_diagonal(X_cov, 2)  # Replace diagonal elements with 2
            sigma_U = 2.0
        else:
            mu_X, mu_U = None, None
            self.covar_list, unmeasured_covar = self.random_select_Twins_confounders()

        ##############################################
        # Sample data per environment
        ##############################################
        data_dict = {}
        ate_dict = {}
        if self.use_twins_covariates is False:
            environment_labels = range(self.n_envs)
        else:
            environment_labels = self.env_idx.keys()

        for i, e in enumerate(environment_labels):

            if self.use_twins_covariates is False:
                # Observed confounders
                X = self.random_state.multivariate_normal(
                    mu_X[i, :],
                    X_cov / np.sqrt(self.n_observed_confounders),
                    size=(n_samples),
                )
                X_representation = x_transform(X)
                # Unmeasured confounder
                U = self.random_state.normal(mu_U[i], sigma_U, size=(n_samples, 1))

            else:
                sample_idx = self.env_idx[e]
                n_samples = len(sample_idx)  # overrides n_samples

                # Observed confounders
                X = np.zeros((n_samples, len(self.covar_list)))
                for j, c in enumerate(self.covar_list):
                    X[:, j] = self.df_covar[c].values[sample_idx]
                    assert len(np.unique(X[:, j])) != 1, f"variable {c} is constant"
                X_representation = x_transform(X)

                # Unmeasured confounder
                if len(unmeasured_covar) > 0:
                    U = np.zeros((n_samples, len(unmeasured_covar)))
                    for j, c in enumerate(unmeasured_covar):
                        U[:, j] = self.df_covar[c].values[sample_idx]
                        assert len(np.unique(U[:, j])) != 1, f"variable {c} is constant"

                else:
                    U = np.zeros((n_samples, 1))

            # Treatment
            treatment_confounding = 1.0 if self.conf_strength != 0.0 else 0.0
            A = (
                intercept_a[:, i]
                + (X_representation @ x_to_a_coef[:, i]).reshape(-1, 1)
                + treatment_confounding * np.sum(np.abs(U), axis=1).reshape(-1, 1)
                + self.random_state.normal(0, 1 / 2, size=(n_samples, 1))
            )

            # Counterfactuals
            Y = (
                self.transportability_violation * intercept_y[:, i]
                + (X_representation @ x_to_y_coef[:, i]).reshape(-1, 1)
                + self.conf_strength * np.sum(np.abs(U), axis=1).reshape(-1, 1)
                + (a_to_y_effect[:, i] * A).reshape(-1, 1)
                + self.random_state.normal(0, 1 / 2, size=(n_samples, 1))
            )

            data_dict[e] = pd.DataFrame(
                np.concatenate([A, Y, X], axis=1),
                columns=["T", "Y"] + self.covar_list,
            )
            ate_dict[e] = a_to_y_effect

        return {"data": data_dict, "ate": ate_dict}
