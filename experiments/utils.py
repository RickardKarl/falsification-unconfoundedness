from itertools import product
from typing import List


def construct_list_of_dicts(original_dict: dict) -> List[dict]:
    """
    Create a list of dictionaries where we iterate
    over the values of the keys that contain lists
    and create a separate dictionary for each
    combination of values.

    Args:
        original_dict (dict): Input dict

    Returns:
        []: list of dicts
    """

    # Find keys with list values
    list_keys = [key for key, value in original_dict.items() if isinstance(value, list)]

    # Generate all combinations of list values
    combinations = product(
        *(
            original_dict[key] if key in list_keys else [original_dict[key]]
            for key in original_dict.keys()
        )
    )

    # Create a list of dictionaries with singular values
    result_list = [
        {key: value for key, value in zip(original_dict.keys(), values)}
        for values in combinations
    ]

    # Do some filtering:
    # 1) if n_confounders_observed and n_confounders_total is present, then remove all instances
    # where n_confounders_total < n_confounders_observed
    if "n_confounders_total" in list_keys and "n_confounders_observed" in list_keys:
        idx_list = []
        for idx, tmp_dict in enumerate(result_list):
            if tmp_dict["n_confounders_total"] < tmp_dict["n_confounders_observed"]:
                idx_list.append(idx)
        for idx in sorted(idx_list, reverse=True):
            del result_list[idx]

    return result_list
