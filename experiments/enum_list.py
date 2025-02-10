import sys

sys.path.append("../")

from enum import Enum

from src.transportability.method import TransportabilityTest
from src.hgic.method import HGIC
from src.ours.method import FalsificationAlgorithm


kci_args = {
    "method": "kci",
    "kernelX": "Gaussian",
    "kernelY": "Gaussian",
    "kernelZ": "Gaussian",
    "approx": False,
    "est_width": "median",
    "use_gp": True,
}
min_test_sample_size = 10
linear_kernel_args = {
    "kernelX": "Linear",
    "kernelY": "Linear",
}


class Methods(Enum):

    FisherZTransportability = TransportabilityTest(
        independence_test_args={"method": "fisherz"}
    )
    KernelTransportability = TransportabilityTest(
        max_sample_size_test=500,
        independence_test_args={
            "method": "kci",
            "kernelX": "Gaussian",
            "kernelY": "Gaussian",
            "kernelZ": "Gaussian",
            "approx": False,
            "est_width": "median",
            "use_gp": False,  # set to False due to exceptionally long runtime otherwise
        },
    )
    FisherZHGIC_fisher = HGIC(
        independence_test_args={"method": "fisherz"},
        method_pval_combination="fisher",
        min_test_sample_size=min_test_sample_size,
    )
    FisherZHGIC_tippett = HGIC(
        independence_test_args={"method": "fisherz"},
        method_pval_combination="tippett",
        min_test_sample_size=min_test_sample_size,
    )
    KernelHGIC_one_test = HGIC(
        independence_test_args=kci_args,
        max_tests=1,
        min_test_sample_size=min_test_sample_size,
    )
    KernelHGIC_fisher = HGIC(
        independence_test_args=kci_args,
        method_pval_combination="fisher",
        min_test_sample_size=min_test_sample_size,
    )
    KernelHGIC_tippett = HGIC(
        independence_test_args=kci_args,
        method_pval_combination="tippett",
        min_test_sample_size=min_test_sample_size,
    )

    FisherZHGIC_capped_fisher = HGIC(
        independence_test_args={"method": "fisherz"},
        method_pval_combination="fisher",
        min_test_sample_size=50,
        max_tests=50,
    )
    FisherZHGIC_capped_tippett = HGIC(
        independence_test_args={"method": "fisherz"},
        method_pval_combination="tippett",
        min_test_sample_size=50,
        max_tests=50,
    )
    KernelHGIC_capped_fisher = HGIC(
        independence_test_args=kci_args,
        method_pval_combination="fisher",
        min_test_sample_size=50,
        max_tests=50,
    )
    KernelHGIC_capped_tippett = HGIC(
        independence_test_args=kci_args,
        method_pval_combination="tippett",
        min_test_sample_size=50,
        max_tests=50,
    )

    FalsificationAlgorithm_linear = FalsificationAlgorithm(
        feature_representation="linear",
        feature_representation_params={"degree": 2},
    )
    FalsificationAlgorithm_quadratic = FalsificationAlgorithm(
        feature_representation="poly",
        feature_representation_params={"degree": 2},
    )
    FalsificationAlgorithm_cubic = FalsificationAlgorithm(
        feature_representation="poly",
        feature_representation_params={"degree": 3},
    )
    FalsificationAlgorithm_linear_bs = FalsificationAlgorithm(
        feature_representation="linear",
        feature_representation_params={"degree": 2},
        n_bootstraps=1000,
    )
    FalsificationAlgorithm_quadratic_bs = FalsificationAlgorithm(
        feature_representation="poly",
        feature_representation_params={"degree": 2},
        n_bootstraps=1000,
    )
    FalsificationAlgorithm_cubic_bs = FalsificationAlgorithm(
        feature_representation="poly",
        feature_representation_params={"degree": 3},
        n_bootstraps=1000,
    )
