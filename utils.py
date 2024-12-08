import jax
import jax.numpy as jnp
from jax import scipy, random
from jaxopt import LBFGS
from jax.scipy.special import gammaln, betaln


def sample_l_inf_ball_batch(key: jax.random.PRNGKey, N: int, d: int) -> jnp.ndarray:
    """
    Samples N points uniformly from the L_infty ball in dimension d.

    Procedure for each sample:
    1. Select a coordinate uniformly from 0 to d-1.
    2. Assign either -1 or 1 to this selected coordinate, each with probability 1/2.
    3. Assign the remaining coordinates uniformly from -1 to 1.

    Args:
        key: A JAX PRNGKey.
        N: Number of samples to generate.
        d: Dimension of the space.

    Returns:
        A JAX array of shape (N, d) where each row is a sampled point from the L_infty ball.
    """
    # Split the key into three for independent random operations
    key_select, key_sign, key_uniform = random.split(key, 3)

    # Step 1: Select N coordinates uniformly from [0, d)
    selected_coords = random.randint(key_select, shape=(N,), minval=0, maxval=d)

    # Step 2: Assign -1 or 1 to each selected coordinate
    signs = (
        random.bernoulli(key_sign, p=0.5, shape=(N,)).astype(jnp.float32) * 2 - 1
    )  # Maps True to 1.0 and False to -1.0

    # Step 3: Sample N x d uniform values from [-1, 1]
    uniform_samples = random.uniform(key_uniform, shape=(N, d), minval=-1.0, maxval=1.0)

    # Prepare batch indices for advanced indexing
    batch_indices = jnp.arange(N)

    # Assign the signs to the selected coordinates for each sample
    sampled_points = uniform_samples.at[batch_indices, selected_coords].set(signs)

    return sampled_points


def median_heuristic(probs):
    """
    Compute the kernel bandwidth using the median heuristic.

    Args:
        probs: Predictions of shape (N, num_classes), probabilities for each class.

    Returns:
        Kernel bandwidth selected using the median heuristic.
    """
    # Compute pairwise distances
    pairwise_distances = jnp.linalg.norm(probs[:, None, :] - probs[None, :, :], axis=2)

    # Extract the upper triangular part of the distance matrix (excluding diagonal)
    distances = pairwise_distances[jnp.triu_indices(pairwise_distances.shape[0], k=1)]

    # Compute the median of the distances
    median_bandwidth = jnp.median(distances)

    return median_bandwidth


from jax.scipy.special import gammaln, betaln


def dirichlet_neg_log_likelihood(alpha, X):

    alpha_0 = jnp.sum(alpha)
    ll_per_sample = (
        gammaln(alpha_0)
        - jnp.sum(gammaln(alpha))
        + jnp.sum((alpha - 1.0) * jnp.log(X), axis=1)
    )
    return -jnp.mean(ll_per_sample)


def fit_dirichlet_alpha(X, maxiter=1000, tol=1e-6):
    N, d = X.shape
    alpha_init = jnp.ones(d)

    # LBFGS solver from jaxopt uses line search for step size adaptation
    solver = LBFGS(fun=dirichlet_neg_log_likelihood, tol=tol, maxiter=maxiter)
    res = solver.run(alpha_init, X)
    return res.params


def test_fit_dirichlet_alpha():
    """
    Tests the fitting of Dirichlet distribution parameters using LBFGS from jaxopt.

    Steps:
    1. Defines true Dirichlet parameters.
    2. Samples synthetic data from the Dirichlet distribution.
    3. Fits the Dirichlet parameters to the sampled data.
    4. Asserts that the fitted parameters are close to the true parameters.
    5. Prints the result of the test.

    Raises:
        AssertionError: If the fitted parameters are not within the specified tolerance.
    """

    def dirichlet_neg_log_likelihood(alpha, X):
        """
        Computes the negative log-likelihood of data X under a Dirichlet distribution with parameters alpha.

        Args:
            alpha (jnp.ndarray): Dirichlet parameters, shape (d,).
            X (jnp.ndarray): Data points on the simplex, shape (N, d).

        Returns:
            float: Negative log-likelihood.
        """
        alpha_0 = jnp.sum(alpha)
        ll_per_sample = (
            gammaln(alpha_0)
            - jnp.sum(gammaln(alpha))
            + jnp.sum((alpha - 1.0) * jnp.log(X), axis=1)
        )
        return -jnp.sum(ll_per_sample)

    def fit_dirichlet_alpha(X, maxiter=1000, tol=1e-6):
        """
        Fits Dirichlet parameters to data X using the LBFGS optimizer.

        Args:
            X (jnp.ndarray): Data points on the simplex, shape (N, d).
            maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-6.

        Returns:
            jnp.ndarray: Fitted Dirichlet parameters, shape (d,).
        """
        _, d = X.shape
        alpha_init = jnp.ones(d)  # Initialize alpha to ones

        # Define the objective function with X fixed using a lambda
        objective = lambda alpha: dirichlet_neg_log_likelihood(alpha, X)

        # Initialize the LBFGS solver
        solver = LBFGS(fun=objective, tol=tol, maxiter=maxiter)

        # Run the solver starting from alpha_init
        res = solver.run(alpha_init)

        return res.params

    def sample_dirichlet(alpha, key, n_samples=1000):
        """
        Samples points from a Dirichlet distribution.

        Args:
            alpha (jnp.ndarray): Dirichlet parameters, shape (d,).
            key (jax.random.PRNGKey): PRNG key for sampling.
            n_samples (int, optional): Number of samples to generate. Defaults to 1000.

        Returns:
            jnp.ndarray: Sampled data points on the simplex, shape (n_samples, d).
        """
        # Sample from Gamma distributions
        gamma_samples = jax.random.gamma(
            key, a=alpha, shape=(n_samples, alpha.shape[0])
        )
        # Normalize to obtain points on the simplex
        return gamma_samples / jnp.sum(gamma_samples, axis=1, keepdims=True)

    # Step 1: Define true Dirichlet parameters
    alpha_true = jnp.array(
        [2.0, 5.0, 3.0]
    )  # Example parameters for a 3-dimensional simplex

    # Step 2: Generate synthetic data
    key = jax.random.PRNGKey(0)  # Seed for reproducibility
    n_samples = 2000  # Number of synthetic data points
    X = sample_dirichlet(alpha_true, key, n_samples)

    # Step 3: Fit Dirichlet parameters to the synthetic data
    alpha_hat = fit_dirichlet_alpha(X)

    # Step 4: Validate the fitted parameters
    tolerance = 0.1  # Define a tolerance for parameter closeness
    max_diff = jnp.max(jnp.abs(alpha_hat - alpha_true))

    assert max_diff < tolerance, (
        f"Fitted alpha {alpha_hat} not close to true alpha {alpha_true}. "
        f"Maximum difference {max_diff} exceeds tolerance {tolerance}."
    )

    # Step 5: Print the test result
    print("Test passed. Fitted alpha is close to the true alpha parameters.")
    print(f"True alpha: {alpha_true}")
    print(f"Fitted alpha: {alpha_hat}")
    print(f"Maximum difference: {max_diff}")


if __name__ == "__main__":
    print("Dirichlet Calibration Error")
    test_fit_dirichlet_alpha()
