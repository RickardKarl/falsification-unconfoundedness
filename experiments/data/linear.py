import numpy as np
import pandas as pd
from scipy.stats import uniform


class LinearDGP):

    param_names = [
        "alpha_0",
        "alpha_X",
        "alpha_U",
        "beta_0",
        "beta_X",
        "beta_U",
        "gamma_T",
        "gamma_X",
        "gamma_U",
        "mu_U",
        "sigma_U",
        "mu_X",
        "sigma_X",
        "sigma_T",
        "sigma_Y",
    ]

    def __init__(
        self,
        n_envs: int = 25,
        conf_strength_non_interaction: float = 1.0,
        conf_strength_interaction: float = 1.0,
        binary_t: bool = False,
        param_dist: dict = {"mu_U": uniform(0, 2)},
        **_,
    ) -> None:
        self.n_envs = n_envs
        self.conf_strength_non_interaction = conf_strength_non_interaction
        self.conf_strength_interaction = conf_strength_interaction
        self.binary_t = binary_t
        self.param_dist = param_dist

    def sample(self, n_samples: int) -> dict:
        return LinearDGP.dgp(
            n_samples,
            self.n_envs,
            self.conf_strength_non_interaction,
            self.conf_strength_interaction,
            binary_treatment=self.binary_t,
            param_dist=self.param_dist,
        )

    def get_covar(self):
        return ["X"]

    def dgp(
        n_samples: int,
        n_envs: int,
        conf_strength_non_interaction: float,
        conf_strength_interaction: float,
        binary_treatment=False,
        param_dist: dict = {},
        seed=None,
    ) -> dict:
        """
        Generate data from a linear causal model
        """

        random_state = np.random.RandomState(seed)

        for p in LinearDGP.param_names:
            if p not in param_dist:
                param_dist[p] = None

        ### Define parameters

        # treatment assigment
        alpha_0 = (
            param_dist["alpha_0"].rvs(n_envs, random_state=random_state)
            if param_dist["alpha_0"]
            else 1 / 2 * np.ones((n_envs))
        )
        alpha_X = (
            param_dist["alpha_X"].rvs(n_envs, random_state=random_state)
            if param_dist["alpha_X"]
            else 1 / 3 * np.ones((n_envs))
        )
        alpha_U = (
            param_dist["alpha_U"].rvs(n_envs, random_state=random_state)
            if param_dist["alpha_U"]
            else 1 / 4 * np.ones((n_envs))
        )

        # control outcome
        beta_0 = (
            param_dist["beta_0"].rvs(n_envs, random_state=random_state)
            if param_dist["beta_0"]
            else 1 / 2 * np.ones((n_envs))
        )
        beta_X = (
            param_dist["beta_X"].rvs(n_envs, random_state=random_state)
            if param_dist["beta_X"]
            else 1 / 3 * np.ones((n_envs))
        )
        beta_U = (
            param_dist["beta_U"].rvs(n_envs, random_state=random_state)
            if param_dist["beta_U"]
            else 1 / 4 * np.ones((n_envs))
        )

        # treatment effect
        gamma_T = (
            param_dist["gamma_T"].rvs(n_envs, random_state=random_state)
            if param_dist["gamma_T"]
            else 1 / 2 * np.ones((n_envs))
        )
        gamma_X = (
            param_dist["gamma_X"].rvs(n_envs, random_state=random_state)
            if param_dist["gamma_X"]
            else 1 / 3 * np.ones((n_envs))
        )
        gamma_U = (
            param_dist["gamma_U"].rvs(n_envs, random_state=random_state)
            if param_dist["gamma_U"]
            else 1 / 4 * np.ones((n_envs))
        )

        # noise
        mu_U = (
            param_dist["mu_U"].rvs(n_envs, random_state=random_state)
            if param_dist["mu_U"]
            else np.ones((n_envs))
        )
        sigma_U = (
            param_dist["sigma_U"].rvs(n_envs, random_state=random_state)
            if param_dist["sigma_U"]
            else np.ones((n_envs))
        )

        mu_X = (
            param_dist["mu_X"].rvs(n_envs, random_state=random_state)
            if param_dist["mu_X"]
            else np.ones((n_envs))
        )
        sigma_X = (
            param_dist["sigma_X"].rvs(n_envs, random_state=random_state)
            if param_dist["sigma_X"]
            else np.ones((n_envs))
        )

        sigma_T = (
            param_dist["sigma_T"].rvs(n_envs, random_state=random_state)
            if param_dist["sigma_T"]
            else 1 / 8 * np.ones((n_envs))
        )

        sigma_Y = (
            param_dist["sigma_Y"].rvs(n_envs, random_state=random_state)
            if param_dist["sigma_Y"]
            else 1 / 8 * np.ones((n_envs))
        )

        ### Sample data from each environment
        data_dict = {}
        ate_dict = {}
        for e in range(n_envs):

            # Unobserved confounder
            U = random_state.normal(mu_U[e], sigma_U[e], size=(n_samples, 1))

            # Observed confounder
            X = random_state.normal(mu_X[e], sigma_X[e], size=(n_samples, 1))

            # Treatment
            treatment_confounding = (
                1.0
                if (conf_strength_non_interaction != 0.0)
                or (conf_strength_interaction != 0.0)
                else 0.0
            )
            t = alpha_0[e] + alpha_X[e] * X + treatment_confounding * alpha_U[e] * U
            if binary_treatment:
                T = random_state.binomial(1, 1 / (1 + np.exp(-t)))
            else:
                T = t + random_state.normal(0, sigma_T[e], size=(n_samples, 1))

            # Counterfactuals
            y = (
                beta_0[e]
                + beta_X[e] * X
                + conf_strength_non_interaction * beta_U[e] * U
            )
            y += T * (
                gamma_T[e] + gamma_X[e] * X + conf_strength_interaction * gamma_U[e] * U
            )

            ate = np.mean(
                gamma_T[e] + gamma_X[e] * X + conf_strength_interaction * gamma_U[e] * U
            )

            # Observed outcome
            Y = y + random_state.normal(0, sigma_Y[e], size=(n_samples, 1))

            data_dict[e] = pd.DataFrame(
                np.concatenate([U, X, T, Y], axis=1), columns=["U", "X", "T", "Y"]
            )
            ate_dict[e] = ate

        return {"data": data_dict, "ate": ate_dict}
