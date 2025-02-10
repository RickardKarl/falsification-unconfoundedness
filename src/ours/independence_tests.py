import numpy as np
import jax.numpy as jnp


###############################################################
# Test based on computing Frobenius norm of off-diagonal block
###############################################################


def compute_offdiag_block_frobnorm(data_x, data_y):

    dim_x, dim_y = data_x.shape[1], data_y.shape[1]
    assert data_x.shape[0] == data_y.shape[0], "first dimension be the same"
    coefs = np.hstack([data_x, data_y])

    validate_matrix(coefs)

    covariance_matrix = np.cov(coefs, rowvar=False)
    offdiag_block = covariance_matrix[:dim_x, dim_x:]
    assert offdiag_block.shape == (dim_x, dim_y)

    return np.linalg.norm(offdiag_block, "fro")


def permutation_independence_test(
    data_x: np.ndarray, data_y: np.ndarray, n_bootstraps: int = 1000, random_state=None
) -> float:

    if random_state is None:
        random_state = np.random.RandomState()

    observed_frob_norm = compute_offdiag_block_frobnorm(data_x, data_y)

    resampled_frob_norm = np.zeros((n_bootstraps, 1))
    for j in range(n_bootstraps):

        # permute rows in coef_t
        permuted_data_x = random_state.permutation(data_x)  # permutates on first axis
        resampled_frob_norm[j] = compute_offdiag_block_frobnorm(permuted_data_x, data_y)

    return np.mean(observed_frob_norm < resampled_frob_norm)


def bootstrapped_permutation_independence_test(
    data_x: np.ndarray,
    data_y: np.ndarray,
    resampled_data_x: np.ndarray,
    resampled_data_y: np.ndarray,
    random_state=None,
) -> float:
    if random_state is None:
        random_state = np.random.RandomState()

    n_bootstraps = resampled_data_x.shape[0]

    assert resampled_data_x.shape[:1] == resampled_data_y.shape[:1]

    observed_frob_norm = compute_offdiag_block_frobnorm(data_x, data_y)

    resampled_frob_norm = np.zeros((n_bootstraps, 1))
    for j in range(n_bootstraps):

        permuted_resampled_data_x = random_state.permutation(
            resampled_data_x[j, :, :].squeeze()
        )

        resampled_frob_norm[j] = compute_offdiag_block_frobnorm(
            permuted_resampled_data_x, resampled_data_y[j, :, :].squeeze()
        )

    return np.mean(observed_frob_norm < resampled_frob_norm)


##########################################
# Utils
##########################################


def validate_matrix(matrix):
    # Assert that the input is a NumPy array
    assert isinstance(matrix, np.ndarray), "Input must be a NumPy array."

    # Assert no NaN values
    assert not jnp.isnan(matrix).any(), f"Matrix contains NaN values: {matrix}"

    # Assert no infinite values
    assert not np.isinf(matrix).any(), "Matrix contains infinite values."

    # Assert proper dimensionality
    assert matrix.ndim == 2, "Matrix must be 2-dimensional."
