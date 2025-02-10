### Import
import os
import sys
from tqdm import tqdm
from datetime import datetime
import json
import logging
import argparse

import pandas as pd
import numpy as np
from utils import construct_list_of_dicts
from enum_list import Methods
from data.dpg import DGP

# Set environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Set experiment parameters")

    parser.add_argument(
        "--method_list",
        nargs="+",
        choices=list(Methods.__members__.keys()),
        required=True,
        help="List of methods",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Config json file name",
        required=True,
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations"
    )

    return parser.parse_args()


if __name__ == "__main__":

    # Get experiment parameters
    args = parse_args()

    ### Create timestamp for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ### Configure the logging system
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG)
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=f"logging/{timestamp}.log",
    )

    logging.info(f"User input: {' '.join(sys.argv)}")

    # logging.info(f"Devices available to JAX: {jax.devices()}")

    logging.info(f"Timestamp for experiment: {timestamp}")

    # Select methods
    method_list = [Methods[m] for m in args.method_list]
    logging.info(f"Loading methods from list: {method_list}")

    ### Load parameters
    logging.info(f"Loading parameters from config file")
    assert os.path.exists(f"configs/{args.experiment}.json")
    with open(f"configs/{args.experiment}.json", "r") as file:
        config_params = json.load(file)
        name_params = list(config_params.keys())

    # dgp_class = DGP[config_params["dgp"]].value
    dgp_string = config_params["dgp"]
    del config_params["dgp"]  # delete as we do not want this in the for loop
    logging.info(f"Selecting DGP: {dgp_string}")

    # Construct a list of all the the possible combinations of parameter values in the dict
    experiment_params = construct_list_of_dicts(config_params)
    logging.info(f"Succesfully loaded config file with parameters {name_params}")

    # Create df for saving results
    df = pd.DataFrame()
    # File path for saving
    filepath = f"results/{args.experiment}-{timestamp}.csv"
    logging.info(f"Save results to {args.experiment}-{timestamp}")

    ### Start running loop
    for p in tqdm(experiment_params):

        logging.info(f"Running experiment with {p}")
        n_samples = p.pop("n_samples")

        for i in range(args.iterations):

            # Sample data
            dgp = DGP(**p)
            sampled = dgp.sample(n_samples)
            data = sampled["data"]
            ate = sampled["ate"]
            observed_covariates = dgp.get_covar()

            for m in method_list:

                name, alg = m.name, m.value

                logging.info(f"Iteration {i} with {name}.")
                start_time = datetime.now()
                out = alg.test(data, observed_covariates=observed_covariates)
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()

                # Estimate bias from omitted variables
                if "ate" in out:
                    estimated_ate = out["ate"]
                    bias = np.mean(
                        np.array(
                            [
                                np.abs(ate[key] - estimated_ate[key])
                                for key in ate
                                if key in estimated_ate
                            ]
                        )
                    )
                else:
                    bias = 0

                # Save results
                res = {
                    "iterations": i,
                    "experiment": args.experiment,
                    "method": m.name,
                    "detection": out["pval"] < 0.05,
                    "bias": bias,
                    "pval": out["pval"],
                    "execution_time": execution_time,
                    **p,
                    "n_samples": n_samples,
                }
                res = pd.DataFrame([res])
                df = pd.concat([df, res], ignore_index=True)
                df.to_csv(filepath, mode="w", index=False)

    logging.info("Finished experiment!")
