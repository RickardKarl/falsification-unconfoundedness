import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import grad, jit, random, lax


def fit_logistic_regression(X, T, alpha=1e-3):
    """
    Fit a logistic regression model using JAX.

    X: Transformed feature matrix (including intercept).
    T: Target variable (binary).
    alpha: Regularization parameter for ridge logistic regression.
    """

    # Define logistic regression loss with optional regularization
    def logistic_loss(params, X, T, alpha):
        logits = X @ params
        loss = jnp.mean(jnp.log(1 + jnp.exp(-T * logits)))  # Logistic loss
        if alpha > 0:
            loss += alpha * jnp.sum(params**2)  # Regularization term (Ridge)
        return loss

    # Initial guess for parameters (weights)
    init_params = jnp.zeros(X.shape[1])

    # Compute the gradient of the loss function
    loss_grad = grad(logistic_loss)

    # Minimize the loss using gradient descent
    def update(params, X, T, alpha, learning_rate=0.1):
        grads = loss_grad(params, X, T, alpha)
        return params - learning_rate * grads

    # Fit the model by iterating updates
    params = init_params
    for _ in range(1000):  # Set max iterations for gradient descent
        params = update(params, X, T, alpha)

    return params


def fit_linear_regression(X, Y, alpha=0):
    """
    Fit a linear regression model using JAX.

    X: Transformed feature matrix.
    Y: Target variable.
    alpha: Regularization parameter for ridge regression.
    """
    # Ridge Regression: Regularized least squares
    I = jnp.eye(X.shape[1])  # Identity matrix for regularization
    I = I.at[-1, -1].set(0)  # Exclude intercept from regularization
    params = solve(X.T @ X + alpha * I, X.T @ Y)
    return params


def cross_val_mse(X, Y, model_fn, num_folds):
    """
    Perform cross-validation and compute the mean squared error.

    X: Transformed feature matrix (including intercept).
    Y: Target variable.
    model_fn: Function to fit the model and return the parameters.
    num_folds: Number of folds for cross-validation.
    alpha: Regularization parameter (used for Ridge regression).
    """
    n = X.shape[0]
    fold_size = n // num_folds
    mse_list = []

    for i in range(num_folds):
        # Split data into training and validation sets
        val_indices = jnp.arange(i * fold_size, (i + 1) * fold_size)
        train_indices = jnp.concatenate(
            [jnp.arange(0, i * fold_size), jnp.arange((i + 1) * fold_size, n)]
        )

        X_train, X_val = X[train_indices], X[val_indices]
        Y_train, Y_val = Y[train_indices], Y[val_indices]

        # Fit the model and get parameters using the training set
        params = model_fn(X_train, Y_train)

        # Compute MSE on validation set
        preds = X_val @ params
        mse = jnp.mean((Y_val - preds) ** 2)
        mse_list.append(mse)

    return jnp.mean(jnp.array(mse_list))


def fit_outcome_model_jax(tf_XT, Y, tf_Xt0, tf_Xt1):

    assert tf_XT.shape[0] > tf_XT.shape[1], "need more samples than features"

    # Fit the outcome model using model_fn
    params_y = fit_linear_regression(tf_XT, Y)

    # Estimate the ATE (Average Treatment Effect)
    ate = jnp.mean(tf_Xt1 @ params_y - tf_Xt0 @ params_y)

    # Perform cross-validation for model diagnostic using the same model_fn
    model_diagnostic = cross_val_mse(tf_XT, Y, fit_linear_regression, num_folds=5)

    return params_y.T, ate, model_diagnostic


def fit_treatment_model_jax(tf_X, T):

    assert tf_X.shape[0] > tf_X.shape[1], "need more samples than features"

    # Fit the model and get parameters
    params_t = fit_linear_regression(tf_X, T, alpha=0)

    return params_t.T


@jit
def bootstrap_model_fitting_jax(Y, T, tf_X, tf_XT, tf_Xt0, tf_Xt1, key):
    # Resample indices using JAX's random module for reproducibility
    key, subkey = random.split(key)  # Split the key to get a new one for resampling

    min_sample_size_needed_for_estimation = tf_X.shape[1] + 1
    assert (
        tf_X.shape[0] > min_sample_size_needed_for_estimation
    ), f"need more samples than {min_sample_size_needed_for_estimation}"
    resampled_indices = resample_until_enough_unique(
        subkey, Y.shape[0], min_sample_size_needed_for_estimation
    )

    # Resample the data
    resampled_Y = Y[resampled_indices]
    resampled_T = T[resampled_indices]
    resampled_tf_X = tf_X[resampled_indices]
    resampled_tf_XT = tf_XT[resampled_indices]
    resampled_tf_Xt0 = tf_Xt0[resampled_indices]
    resampled_tf_Xt1 = tf_Xt1[resampled_indices]

    # Fit outcome and treatment models on resampled data
    resampled_params_y, _, _ = fit_outcome_model_jax(
        resampled_tf_XT, resampled_Y, resampled_tf_Xt0, resampled_tf_Xt1
    )
    resampled_params_t = fit_treatment_model_jax(resampled_tf_X, resampled_T)

    return resampled_params_y, resampled_params_t


def resample_until_enough_unique(subkey, n_resamples, min_sample_size):
    # Initial resampling
    resampled_indices = random.choice(
        subkey, n_resamples, shape=(n_resamples,), replace=True
    )

    def count_unique(x):
        x = jnp.sort(x)
        return 1 + (x[1:] != x[:-1]).sum()

    # Define condition function for while loop
    def condition_fn(state):
        _, resampled_indices = state
        # Check if unique indices are below the threshold
        return count_unique(resampled_indices) < min_sample_size

    # Define body function for while loop
    def body_fn(state):
        subkey, _ = state
        # Resample and update state
        subkey, new_subkey = random.split(subkey)
        resampled_indices = random.choice(
            new_subkey, n_resamples, shape=(n_resamples,), replace=True
        )
        return (subkey, resampled_indices)

    # Initial state: (key, resampled_indices)
    state = (subkey, resampled_indices)

    # Apply while loop until the condition is met
    _, resampled_indices = lax.while_loop(condition_fn, body_fn, state)

    return resampled_indices
