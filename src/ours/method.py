import numpy as np
import pandas as pd
from jax import random, vmap, config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

from src.abstract import AbstractAlgorithm
from src.ours.utils import create_polynomial_representation
from src.ours.linear_regression_jax import (
    bootstrap_model_fitting_jax,
    fit_outcome_model_jax,
    fit_treatment_model_jax,
)
from src.ours.independence_tests import (
    permutation_independence_test,
    bootstrapped_permutation_independence_test,
)


class FalsificationAlgorithm(AbstractAlgorithm):

    def __init__(
        self,
        feature_representation: str = "linear",
        feature_representation_params: dict = {},
        ridge_regression: bool = False,
        binary_treatment: bool = False,
        min_samples_per_env: int = 25,
        independence_test_args: dict = {},
        n_bootstraps: int = 1000,
    ) -> None:
        """_summary_

        Args:
            feature_representation (str, optional): feature representatin to use. Defaults to "linear".
            feature_representation_params (dict, optional): params forwarded to feature representation. Defaults to {}.
            ridge_regression (bool, optional): whether to use L1 regularization. Defaults to False.
             binary_treatment (bool, optional): whether we assume binary or cont. treatment. Defaults to False.
            min_samples_per_env (int, optional): minimum environments needed per environment. Defaults to 25.
            independence_test_args (dict, optional): arguments for CIT method. Defaults to {}.
            n_bootstraps (int, optional): number of bootstraps. Defaults to 1000. If None, then no bootsstrap is used.

        """
        self.feature_representation = feature_representation
        self.feature_representation_params = feature_representation_params
        self.ridge_regression = ridge_regression
        self.min_samples_per_env = min_samples_per_env
        self.binary_treatment = binary_treatment
        self.independence_test_args = independence_test_args
        self.n_bootstraps = n_bootstraps

        if self.ridge_regression is True or self.binary_treatment is True:
            raise NotImplementedError

    def test(self, data: dict, observed_covariates: list) -> dict:
        """
        Args:
            data (dict): Dictionary of dataframes with keys corresponding to environments.
            observed_covariates (list): List of observed covariates.

        Returns:
            dict: Dictionary with output of the test.
        """

        n_environments = len(data)
        ate_dict = {}  # To store ATE estimates
        outcome_model_diagnostics = {}  # To store model fit metric for outcome model
        coef_model_y, coef_model_t = [], []  # To store coefficients for models
        resampled_coef_model_y, resampled_coef_model_t = [], []

        # Get feature representations
        all_X = np.row_stack(
            [data[env][observed_covariates].to_numpy() for env in data]
        )
        all_T = np.row_stack([data[env]["T"].to_numpy().reshape(-1, 1) for env in data])
        phi_x = self.get_feature_representation(all_X)
        phi_xt = self.get_feature_representation(np.concatenate([all_X, all_T], axis=1))

        for key, env_dataset in data.items():
            if not self.validate_data(key, env_dataset):
                n_environments -= 1
                continue

            Y = env_dataset["Y"].to_numpy().reshape(-1, 1)
            T = env_dataset["T"].to_numpy().reshape(-1, 1)
            X = (
                env_dataset[observed_covariates]
                .to_numpy()
                .reshape(-1, len(observed_covariates))
            )

            # Cast data to jax.numpy
            Y = jnp.array(Y)
            T = jnp.array(T)
            X = jnp.array(X)

            add_intercept = lambda term: jnp.hstack(
                [term, jnp.ones((term.shape[0], 1))]
            )
            # Precompute the features (with intercept term) for all input data
            tf_X = add_intercept(phi_x(X))
            tf_XT = add_intercept(phi_xt(jnp.concatenate([X, T], axis=1)))
            tf_Xt1 = add_intercept(
                phi_xt(jnp.concatenate([X, jnp.ones(T.shape)], axis=1))
            )
            tf_Xt0 = add_intercept(
                phi_xt(jnp.concatenate([X, jnp.zeros(T.shape)], axis=1))
            )

            params_y, ate, model_diagnostic_y = fit_outcome_model_jax(
                tf_XT, Y, tf_Xt0, tf_Xt1
            )
            coef_model_y.append(params_y)
            ate_dict[key] = ate
            outcome_model_diagnostics[key] = model_diagnostic_y

            params_t = fit_treatment_model_jax(tf_X, T)
            coef_model_t.append(params_t)

            if self.n_bootstraps:

                keys = random.split(random.PRNGKey(0), self.n_bootstraps)

                # Vectorize the model fitting across bootstrap iterations
                resampled_params = vmap(
                    bootstrap_model_fitting_jax,
                    in_axes=(
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        0,
                    ),  # Ensure each argument has the correct shape
                )(
                    Y,
                    T,
                    tf_X,
                    tf_XT,
                    tf_Xt0,
                    tf_Xt1,
                    keys,
                )

                # Append the results
                resampled_coef_model_y.append(resampled_params[0])
                resampled_coef_model_t.append(resampled_params[1])

        # Cast back to numpy for the indepencene test
        coef_model_y = np.array(jnp.vstack(coef_model_y))  # (n_envs, dim_y)
        coef_model_t = np.array(jnp.vstack(coef_model_t))  # (n_envs, dim_t)

        if self.n_bootstraps:
            resampled_coef_model_y = np.array(
                jnp.hstack(resampled_coef_model_y)
            )  # (n_bootstraps, n_envs, dim_y)
            resampled_coef_model_t = np.array(
                jnp.hstack(resampled_coef_model_t)
            )  # (n_bootstraps, n_envs, dim_t)

            # Run independence test using the coefficients from models
            pval = self.run_bootstrapped_independence_test(
                coef_model_y,
                coef_model_t,
                resampled_coef_model_y,
                resampled_coef_model_t,
            )
        else:
            pval = self.run_independence_test(coef_model_y, coef_model_t)

        return {
            "pval": pval,
            "ate": ate_dict,
            "outcome_model_diagnostics": outcome_model_diagnostics,
            "coef_model_y": np.asarray(coef_model_y).squeeze(),
            "coef_model_t": np.asarray(coef_model_t).squeeze(),
        }

    def get_feature_representation(self, features):

        if self.feature_representation == "linear":
            return lambda x: x

        elif self.feature_representation == "poly":
            return lambda x: create_polynomial_representation(
                x, **self.feature_representation_params
            )

        else:
            raise ValueError(
                f"Invalid choice of feature representation: {self.feature_representation}"
            )

    def validate_data(self, key: str, env_dataset: pd.DataFrame) -> bool:

        assert "T" in env_dataset, f"Key {key} does not contain treatment variable T"
        assert "Y" in env_dataset, f"Key {key} does not contain outcome variable Y"

        if len(env_dataset) < self.min_samples_per_env:
            return False
        return True

    def run_independence_test(self, data_x, data_y):
        return permutation_independence_test(
            data_x=data_x,
            data_y=data_y,
        )

    def run_bootstrapped_independence_test(
        self, data_x, data_y, resampled_data_x, resampled_data_y
    ):
        return bootstrapped_permutation_independence_test(
            data_x=data_x,
            data_y=data_y,
            resampled_data_x=resampled_data_x,
            resampled_data_y=resampled_data_y,
        )
