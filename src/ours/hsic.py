import jax.numpy as jnp
from jax import random
import jax
import numpy as np


from itertools import permutations, combinations, chain


@jax.jit
def compute_biased_hsic(gram_U: jnp.ndarray, gram_V: jnp.ndarray) -> float:
    """
    Compute Hilbert-Schmidt Independence Criterion between two gram matrices

    Args:
        gram_U (jnp.ndarray): gram matrix of shape (N, N)
        gram_V (jnp.ndarray): gram matrix of shape (N, N)

    Returns:
        float: returns N*HSIC
    """
    assert gram_U.shape == gram_V.shape, "Gram matrices must be of same shape"

    N = gram_U.shape[0]

    # Compute centering matrix
    H = jnp.eye(N) - 1 / N * jnp.ones((N, N))

    # Compute biased HSIC
    biased_HSIC = (1 / N) * jnp.trace(gram_U @ H @ gram_V @ H)

    return biased_HSIC


@jax.jit
def compute_normalized_hsic(gram_U: jnp.ndarray, gram_V: jnp.ndarray) -> float:
    """
    Compute a normalized Hilbert-Schmidt Independence Criterion between two gram matrices

    Args:
        gram_U (jnp.ndarray): gram matrix of shape (N, N)
        gram_V (jnp.ndarray): gram matrix of shape (N, N)

    Returns:
        float: returns normalized HSIC
    """
    assert gram_U.shape == gram_V.shape, "Gram matrices must be of same shape"

    unnorm_hsic = compute_biased_hsic(gram_U, gram_V)

    hsic_U = compute_biased_hsic(gram_U, gram_U)
    hsic_V = compute_biased_hsic(gram_V, gram_V)

    return unnorm_hsic / jnp.sqrt(hsic_U * hsic_V)


@jax.jit
def compute_squared_norm_mean_embedding(gram_matrix: jnp.ndarray) -> float:
    """
    Computes the estimate of the squared norm of the kernel mean embedding of
    the distribution generating the data given the gram matrix of the data

    Args:
        gram_matrix (jnp.ndarray): gram matrix of shape (N, N)

    Returns:
        float: estimate of the squared norm of the kernel mean embedding
    """

    N = gram_matrix.shape[0]
    # Vectorized code
    ordered_combinations_array = jnp.array(ordered_combinations(N, 2))
    squared_norm = (1 / n_permute_m(N, 2)) * jnp.sum(
        gram_matrix[ordered_combinations_array[:, 0], ordered_combinations_array[:, 1]]
    )

    return squared_norm


@jax.jit
def compute_expected_biased_hsic(gram_U: jnp.ndarray, gram_V: jnp.ndarray) -> float:
    """
    Computes the estimated expected value of the HSIC between two gram matrices
    """

    assert gram_U.shape == gram_V.shape, "Gram matrices must be of same shape"

    N = gram_U.shape[0]

    squared_norm_mu_u = compute_squared_norm_mean_embedding(gram_U)
    squared_norm_mu_v = compute_squared_norm_mean_embedding(gram_V)

    expected_biased_hsic = (
        1
        / N
        * (
            1
            + squared_norm_mu_u * squared_norm_mu_v
            - squared_norm_mu_u
            - squared_norm_mu_v
        )
    )

    return expected_biased_hsic


@jax.jit
def compute_variance_biased_hsic(
    gram_U: jnp.ndarray, gram_V: jnp.ndarray, H: jnp.ndarray
) -> float:
    """
    Computes the estimated biased variance of the HSIC between two gram matrices

    Args:
        gram_U (jnp.ndarray): gram matrix for U of shape (N, N)
        gram_V (jnp.ndarray): gram matrix for V of shape (N, N)
        H (jnp.ndarray): centering matrix of shape (N, N)

    Returns:
        float: estimated variance of HSIC

    """
    assert gram_U.shape == gram_V.shape, "Gram matrices must be of same shape"

    # number of samples
    N = gram_U.shape[0]
    vec_ones = jnp.ones((N, 1))
    # centering matrix
    H = jnp.eye(N) - 1 / N * jnp.ones((N, N))

    # See definition of B in Theorem 4 in Gretton et al. (2008)
    B = jnp.power((H @ gram_U @ H) * (H @ gram_V @ H), 2)
    variance_biased_hsic = (
        2
        * (N - 4)
        * (N - 5)
        / (n_permute_m(N, 4) * N * (N - 1))
        * vec_ones.T
        @ (B - jnp.diag(jnp.diag(B)))
        @ vec_ones
    ).flatten()[0]

    return variance_biased_hsic


def n_permute_m(m: int, n: int) -> float:
    """
    Computes the number of arrangement of n items from m objects, mPn, equal to
    m!/(m-n)!.

    Args:
        m (int): number of objects
        n (int): number of items

    Returns:
        float: number of arrangements
    """
    assert 0 < n <= m

    return np.cumprod(range(m - n + 1, m + 1))[-1]


def ordered_combinations(m: int, n: int) -> list:
    """
    Generates a list of all ordered combinations of n elements from a set of m
    objects. For example, if m = 3 and n = 2, the result is
    [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)].

    Args:
        m (int): number of objects
        n (int): number of items

    Returns:
        list: list of all ordered combinations

    """
    l_ordered_combinations = [
        list(permutations(elt)) for elt in combinations(range(m), n)
    ]
    l_ordered_combinations = list(chain.from_iterable(l_ordered_combinations))

    return l_ordered_combinations


# @jax.jit
def linear_kernel(x: jnp.ndarray, y: jnp.ndarray = None) -> jnp.array:

    if y is None:
        kmatrix = jnp.matmul(x, x.T)

    else:
        kmatrix = jnp.matmul(x, y.T)

    assert kmatrix.shape[0] == kmatrix.shape[1]
    return kmatrix
