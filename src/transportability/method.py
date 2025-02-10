from src.abstract import AbstractAlgorithm

import numpy as np
from causallearn.utils.cit import CIT
from numpy.linalg import LinAlgError


class TransportabilityTest(AbstractAlgorithm):

    default_independence_test_args = {
        "kernelX": "Gaussian",
        "kernelY": "Gaussian",
        "approx": True,
        "est_width": "median",
    }

    def __init__(
        self, independence_test_args={}, max_sample_size_test: int = np.inf
    ) -> None:

        self.independence_test_args = independence_test_args
        self.max_sample_size_test = max_sample_size_test
        for (
            key,
            item,
        ) in TransportabilityTest.default_independence_test_args.items():
            if key not in self.independence_test_args:
                self.independence_test_args[key] = item

    def test(self, data: dict, observed_covariates: list) -> dict:

        # Initialize lists to store results
        Y_list, T_list, S_list, X_list = [], [], [], []

        # Process each environment and its corresponding DataFrame
        for env, df in data.items():

            # Create environment indicator
            env_indicator = np.full((len(df), 1), env)
            S_list.append(env_indicator)

            # Extract Y and X values
            Y_list.append(df["Y"].values.reshape(-1, 1))
            T_list.append(df["T"].values.reshape(-1, 1))
            X_list.append(df[observed_covariates].values)

        # Concatenate lists into numpy arrays with correct shapes
        Y = np.vstack(Y_list)  # Shape: (total_samples, 1)
        T = np.vstack(T_list)  # Shape: (total_samples, 1)
        S = np.vstack(S_list)  # Shape: (total_samples, 1)
        X = np.vstack(X_list)  # Shape: (total_samples, num_covariates)

        # Subsample data if necessary
        if Y.shape[0] > self.max_sample_size_test:
            Y, S, X, T = self.subsample_data(Y, S, X, T)

        return self.run_independence_test(Y, S, X, T)

    def subsample_data(self, y, s, x, t):

        unique_s, counts = np.unique(s, return_counts=True)
        proportions = counts / counts.sum()

        sampled_indices = []
        for s_value, proportion in zip(unique_s, proportions):
            s_indices = np.where(s.flatten() == s_value)[0]
            n_samples = min(
                len(s_indices),
                int(np.round(proportion * self.max_sample_size_test)),
            )
            sampled_indices.extend(
                np.random.choice(s_indices, n_samples, replace=False)
            )

        # Ensure sampled_indices has no more than self.max_sample_size_test
        if len(sampled_indices) > self.max_sample_size_test:
            sampled_indices = np.random.choice(
                sampled_indices, self.max_sample_size_test, replace=False
            )

        y = y[sampled_indices, :]
        s = s[sampled_indices, :]
        x = x[sampled_indices, :]
        t = t[sampled_indices, :]

        return y, s, x, t

    def run_independence_test(self, y, s, x, t):

        data_combined = np.hstack([y, s, x, t])
        try:
            cit_obj = CIT(data_combined, **self.independence_test_args)
            pval = cit_obj(0, 1, list(range(2, data_combined.shape[1])))
        except LinAlgError as e:
            print("An error occurred: ", e)
            pval = 1.0
        return {"pval": pval}
