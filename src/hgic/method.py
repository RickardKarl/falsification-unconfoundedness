from re import I
import warnings

from src.abstract import AbstractAlgorithm

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues
from causallearn.utils.cit import CIT
from numpy.linalg import LinAlgError
from scipy.stats import zscore

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)


class HGIC(AbstractAlgorithm):

    def __init__(
        self,
        independence_test_args: dict,
        max_tests: int = -1,
        min_test_sample_size: int = 25,
        method_pval_combination: str = "fisher",
    ) -> None:

        self.independence_test_args = independence_test_args
        self.max_tests = max_tests
        self.min_test_sample_size = min_test_sample_size
        self.method_pval_combination = method_pval_combination

        if "feature_scaling" in self.independence_test_args:
            self.feature_scaling = self.independence_test_args.pop("feature_scaling")
        else:
            self.feature_scaling = False

    def test(self, data: dict, observed_covariates=[]) -> dict:
        # Match samples columns into pair with about the same sample size

        # Sort environments w.r.t to sample size
        sample_sizes = [(key, len(data[key])) for key in data]
        sample_sizes.sort(key=lambda y: y[1], reverse=True)
        # this is the number of samples in the second largest environment,
        # which is the maximum number of samples we can use to test
        max_nbr_samples = sample_sizes[1][1]

        def get_samples(key, var):
            return data[key][var].values

        if self.max_tests < 1:
            max_tests = max_nbr_samples  # ensures we use all samples to test
        else:
            max_tests = self.max_tests

        df_list = []
        n = 0
        while n < max_nbr_samples - 1:
            tmp_key_list = [key for key, ns in sample_sizes if ns > n + 1]

            if len(tmp_key_list) < self.min_test_sample_size:
                n += 2
                continue

            # Select first sample in each environment
            T_i = np.array([get_samples(key, "T")[n] for key in tmp_key_list])
            Y_i = np.array([get_samples(key, "Y")[n] for key in tmp_key_list])

            # Select second sample in each environment
            T_j = np.array([get_samples(key, "T")[n + 1] for key in tmp_key_list])
            Y_j = np.array([get_samples(key, "Y")[n + 1] for key in tmp_key_list])

            data_dict = {"Y_i": Y_i, "T_i": T_i, "T_j": T_j, "Y_j": Y_j}

            # Select observed confounders
            for var_name in observed_covariates:
                X_i = np.array([get_samples(key, var_name)[n] for key in tmp_key_list])
                X_j = np.array(
                    [get_samples(key, var_name)[n + 1] for key in tmp_key_list]
                )
                data_dict[f"{var_name}_i"] = X_i
                data_dict[f"{var_name}_j"] = X_j

            # Collect data in dict to save it as a dataframe
            df_list.append(pd.DataFrame(data=data_dict))

            n += 2

            if len(df_list) >= max_tests:
                break

        # Add covariates to the list of conditional variables
        cond_var = []
        for var_name in observed_covariates:
            cond_var.append(f"{var_name}_i")
            cond_var.append(f"{var_name}_j")

        # Perform statistical test for each pair of samples

        pval_list = []
        number_samples_used = []
        for df_tmp in df_list:
            res = self.statistical_test(df_tmp, "T_j", "Y_i", cond_var + ["T_i"])

            if np.isnan(res["pval"]):
                print("computed nan p-value")
            else:
                pval_list.append(res["pval"])
                number_samples_used.append(len(df_tmp))

        assert len(pval_list) > 0, "No p-values computed"

        # Combine p-values from hypothesis tests with the same null hypothesis
        if self.method_pval_combination == "stouffer":
            # Compute weights normalized by the samples used by each test
            weights = [np.sqrt(n) for n in number_samples_used]
        else:
            weights = None

        _, pval = combine_pvalues(
            pval_list, method=self.method_pval_combination, weights=weights
        )

        return {
            "pval": pval,
            "pval_list": pval_list,
            "number_samples_used": number_samples_used,
        }

    def statistical_test(self, df: pd.DataFrame, X: str, Y: str, Z: list) -> float:

        data = df.to_numpy()
        if self.feature_scaling is True:
            data = zscore(data, axis=0, ddof=1)

        # construct a CIT instance with data and method name
        try:
            cit_obj = CIT(data, **self.independence_test_args)
            pval = cit_obj(
                df.columns.get_loc(X),
                df.columns.get_loc(Y),
                [df.columns.get_loc(z) for z in Z],
            )
        except LinAlgError as e:
            print("An error occurred: ", e)
            pval = 1.0

        return {"pval": pval}
