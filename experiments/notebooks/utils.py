import sys

sys.path.append("../../")

from experiments.enum_list import Methods

names = {
    "FalsificationAlgorithm_linear": "Ours without bootstrap (Linear)",
    "FalsificationAlgorithm_quadratic": "Ours without bootstrap (Quadratic)",
    "FalsificationAlgorithm_cubic": "Ours without bootstrap (Cubic)",
    "FalsificationAlgorithm_linear_bs": "Ours (Linear)",
    "FalsificationAlgorithm_quadratic_bs": "Ours (Quadratic)",
    "FalsificationAlgorithm_cubic_bs": "Ours (Cubic)",
    "FisherZHGIC_fisher": "HGIC (Pearson)",
    "FisherZHGIC_tippett": "HGIC (Pearson-Tippett)",
    "KernelHGIC_one_test": "HGIC (KCIT-1)",
    "KernelHGIC_fisher": "HGIC (KCIT-Fisher)",
    "KernelHGIC_tippett": "HGIC (KCIT)",
    "FisherZHGIC_capped_fisher": "HGIC  (Pearson-Fisher)",
    "FisherZHGIC_capped_tippett": "HGIC  (Pearson-Tippett)",
    "KernelHGIC_capped_fisher": "HGIC  (KCIT-Fisher)",
    "KernelHGIC_capped_tippett": "HGIC  (KCIT)",
    "FisherZTransportability": "Transp. test (Pearson)",
    "KernelTransportability": "Transp. test (KCIT)",
}


name_order = [
    names["FalsificationAlgorithm_linear"],
    names["FalsificationAlgorithm_quadratic"],
    names["FalsificationAlgorithm_cubic"],
    names["FalsificationAlgorithm_linear_bs"],
    names["FalsificationAlgorithm_quadratic_bs"],
    names["FalsificationAlgorithm_cubic_bs"],
    names["FisherZTransportability"],
    names["KernelTransportability"],
    names["FisherZHGIC_fisher"],
    names["FisherZHGIC_tippett"],
    names["FisherZHGIC_capped_fisher"],
    names["FisherZHGIC_capped_tippett"],
    names["KernelHGIC_one_test"],
    names["KernelHGIC_fisher"],
    names["KernelHGIC_tippett"],
    names["KernelHGIC_capped_fisher"],
    names["KernelHGIC_capped_tippett"],
]

name_shapes = {
    names["FalsificationAlgorithm_linear"]: ">",
    names["FalsificationAlgorithm_quadratic"]: ">",
    names["FalsificationAlgorithm_cubic"]: ">",
    names["FalsificationAlgorithm_linear_bs"]: "h",
    names["FalsificationAlgorithm_quadratic_bs"]: "h",
    names["FalsificationAlgorithm_cubic_bs"]: "h",
    names["FisherZTransportability"]: "d",
    names["KernelTransportability"]: "d",
    names["FisherZHGIC_fisher"]: "s",
    names["FisherZHGIC_tippett"]: "s",
    names["FisherZHGIC_capped_fisher"]: "s",
    names["FisherZHGIC_capped_tippett"]: "s",
    names["KernelHGIC_one_test"]: "^",
    names["KernelHGIC_fisher"]: "^",
    names["KernelHGIC_tippett"]: "^",
    names["KernelHGIC_capped_fisher"]: "^",
    names["KernelHGIC_capped_tippett"]: "^",
}

name_colors = {
    names["FalsificationAlgorithm_linear"]: "purple",
    names["FalsificationAlgorithm_quadratic"]: "purple",
    names["FalsificationAlgorithm_cubic"]: "purple",
    names["FalsificationAlgorithm_linear_bs"]: "purple",
    names["FalsificationAlgorithm_quadratic_bs"]: "purple",
    names["FalsificationAlgorithm_cubic_bs"]: "purple",
    names["FisherZTransportability"]: "g",
    names["KernelTransportability"]: "g",
    names["FisherZHGIC_fisher"]: "blue",
    names["FisherZHGIC_tippett"]: "blue",
    names["FisherZHGIC_capped_fisher"]: "blue",
    names["FisherZHGIC_capped_tippett"]: "blue",
    names["KernelHGIC_one_test"]: "#8F00FF",  # violet
    names["KernelHGIC_fisher"]: "#8F00FF",
    names["KernelHGIC_tippett"]: "#8F00FF",
    names["KernelHGIC_capped_fisher"]: "#8F00FF",
    names["KernelHGIC_capped_tippett"]: "#8F00FF",
}

name_linestyles = {
    names["FalsificationAlgorithm_linear"]: (3, 2, 1, 2),
    names["FalsificationAlgorithm_quadratic"]: (3, 2, 1, 2),
    names["FalsificationAlgorithm_cubic"]: (3, 2, 1, 2),
    names["FalsificationAlgorithm_linear_bs"]: (3, 2, 1, 2),
    names["FalsificationAlgorithm_quadratic_bs"]: (3, 2, 1, 2),
    names["FalsificationAlgorithm_cubic_bs"]: (3, 2, 1, 2),
    names["FisherZTransportability"]: (3, 2, 1, 2),  # Dash-dot line
    names["KernelTransportability"]: (3, 2, 1, 2),  # Dash-dot line
    names["FisherZHGIC_fisher"]: (1, 1),
    names["FisherZHGIC_tippett"]: (1, 1),
    names["FisherZHGIC_capped_fisher"]: (1, 1),
    names["FisherZHGIC_capped_tippett"]: (1, 1),
    names["KernelHGIC_one_test"]: (1, 1),
    names["KernelHGIC_fisher"]: (1, 1),
    names["KernelHGIC_tippett"]: (1, 1),
    names["KernelHGIC_capped_fisher"]: (1, 1),
    names["KernelHGIC_capped_tippett"]: (1, 1),
}


for key in names.keys():
    if key not in name_order:
        assert f"{key} missing from name_order"

    if key not in name_shapes.keys():
        assert f"{key} missing from name_shapes"

    if key not in name_colors.keys():
        assert f"{key} missing from name_colors"

    if key not in name_linestyles.keys():
        assert f"{key} missing from name_linestyles"

# Check if all enums from Methods are in dicts
missing_keys = [method.name for method in Methods if method.name not in names]
assert len(missing_keys) == 0, f"Missing keys from Methods: {missing_keys}"
