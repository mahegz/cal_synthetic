# A collection of calibration errors that rely on nadaray-watson interoplation.
import jax
import jax.numpy as jnp
from jax import scipy, random
from jaxopt import LBFGS
from jax.scipy.special import gammaln, betaln
from utils import fit_dirichlet_alpha


def nadaraya_watson_ece(probs, labels, kernel_fn, p=1):
    """
    Compute the Expected Calibration Error (ECE) using a Nadaraya-Watson estimator.

    Args:
        probs: JAX array of shape (N, K), the predicted probabilities or features.
        labels: JAX array of shape (N,), the true labels.
        kernel_fn: Callable, a kernel function that takes (f_xj, f_xi) and returns kernel weights.
        p: The p-norm to compute the calibration error.

    Returns:
        The Expected Calibration Error (scalar).
    """
    N, K = probs.shape

    # One-hot encode the labels for conditional expectation
    labels_onehot = jax.nn.one_hot(labels, num_classes=K)

    def cond_exp_for_point(f_xj):
        """
        Compute the conditional expectation for a given point f_xj.

        Args:
            f_xj: JAX array of shape (K,), the reference point.

        Returns:
            JAX array, the conditional expectation.
        """
        # Vectorize the kernel function to handle batch computations
        vectorized_kernel = jax.vmap(lambda f_xi: kernel_fn(f_xj, f_xi), in_axes=0)
        kernel_weights = vectorized_kernel(probs)

        # Avoid division by zero
        denom = jnp.sum(kernel_weights)
        denom = jnp.maximum(denom, 1e-10)

        # Compute conditional expectation
        cond_exp = jnp.sum(kernel_weights[:, None] * labels_onehot, axis=0) / denom
        return cond_exp

    # Vectorized computation of conditional expectations
    cond_exp_fn = jax.vmap(cond_exp_for_point, in_axes=0)
    cond_exps = cond_exp_fn(probs)

    # Compute the p-norm calibration error
    ece = jnp.sum(jnp.linalg.norm(cond_exps - probs, ord=p, axis=1) ** p)
    return (ece / N) ** (1 / p)


def log_dirichlet_kernel(f_xj, f_xi, bandwidth):
    """
    Compute the Dirichlet kernel for the probability simplex.

    Args:
        f_xj: JAX array of shape (K,), the reference probability vector.
        f_xi: JAX array of shape (N, K), the probability vectors to compare against.
        bandwidth: Kernel bandwidth.

    Returns:
        JAX array of shape (N,), the Dirichlet kernel values for f_xj.
    """
    alphas = f_xi / bandwidth + 1
    log_beta = jnp.sum(gammaln(alphas), axis=1) - gammaln(jnp.sum(alphas, axis=1))
    log_kernel = jnp.dot(jnp.log(f_xj), (alphas - 1).T) - log_beta
    return log_kernel


def conditional_expectation_dirichlet(f_xj, probs, labels_onehot, bandwidth):
    """
    Compute the conditional expectation for a given probability vector f_xj.

    Args:
        f_xj: JAX array of shape (K,), the reference probability vector.
        probs: JAX array of shape (N, K), the probability vectors to compare against.
        labels_onehot: JAX array of shape (N, K), one-hot encoded labels.
        bandwidth: Kernel bandwidth.

    Returns:
        JAX array of shape (K,), the conditional expectation E[y | f(x)].
    """
    log_kernels = log_dirichlet_kernel(f_xj, probs, bandwidth)
    kernels = jnp.exp(log_kernels)

    # Avoid division by zero
    denom = jnp.sum(kernels)
    denom = jnp.maximum(denom, 1e-10)

    # Compute conditional expectations
    cond_exp = jnp.sum(kernels[:, None] * labels_onehot, axis=0) / denom
    return cond_exp


vec_conditional_expectation_dirichlet = jax.vmap(
    conditional_expectation_dirichlet, (0, None, None, None), (0)
)


def dirichlet_calibration_error(probs, labels, bandwidth, p=1):
    """
    Compute the Dirichlet Calibration Error (DCE) with vmap.

    Args:
        probs: JAX array of shape (N, K), the predicted probabilities.
        labels: JAX array of shape (N,), the true labels.
        bandwidth: Kernel bandwidth.
        p: The p-norm to compute the calibration error.

    Returns:
        The Dirichlet Calibration Error (scalar).
    """
    N, K = probs.shape
    labels_onehot = jax.nn.one_hot(labels, num_classes=K)
    cond_exps = vec_conditional_expectation_dirichlet(
        probs, probs, labels_onehot, bandwidth
    )

    # Compute the p-norm calibration error
    dce = jnp.sum(jnp.linalg.norm(cond_exps - probs, ord=p, axis=1) ** p)
    return (dce / N) ** (1 / p)


def beta_kernel(c_j, c_i, bandwidth):
    """
    Compute the Beta kernel for 1D confidence values.

    Args:
        c_j: JAX array of shape (M,), the reference confidence values.
        c_i: JAX array of shape (N,), the confidence values to compare against.
        bandwidth: Kernel bandwidth.

    Returns:
        JAX array of shape (M, N), the Beta kernel values.
    """
    alpha = c_i / bandwidth + 1
    beta = (1 - c_i) / bandwidth + 1

    log_beta = betaln(alpha, beta)
    log_kernel = (
        (alpha - 1) * jnp.log(c_j[:, None])
        + (beta - 1) * jnp.log(1 - c_j[:, None])
        - log_beta
    )
    return log_kernel


def compute_conditional_expectation_beta(c_j, confidences, labels, bandwidth):
    """
    Compute the conditional expectation for given reference confidence values.

    Args:
        c_j: JAX array of shape (M,), the reference confidence values.
        confidences: JAX array of shape (N,), the predicted confidence values.
        labels: JAX array of shape (N,), the true binary labels.
        bandwidth: Kernel bandwidth.

    Returns:
        JAX array of shape (M,), the conditional expectation E[y | c].
    """
    log_kernels = beta_kernel(c_j, confidences, bandwidth)
    kernels = jnp.exp(log_kernels)

    # Avoid division by zero
    denom = jnp.sum(kernels, axis=1, keepdims=True)
    denom = jnp.maximum(denom, 1e-10)

    # Compute conditional expectations
    cond_exp = jnp.sum(kernels * labels[None, :], axis=1) / denom.squeeze()
    return cond_exp


def beta_calibration_error(confidences, labels, bandwidth, p=1):
    """
    Compute the Beta Calibration Error (BCE) for 1D confidence values.

    Args:
        confidences: JAX array of shape (N,), the predicted confidence values.
        labels: JAX array of shape (N,), the true binary labels (0 or 1).
        bandwidth: Kernel bandwidth.
        p: The p-norm to compute the calibration error.

    Returns:
        The Beta Calibration Error (scalar).
    """
    N = confidences.shape[0]

    # Vectorized computation of conditional expectations using vmap
    cond_exp_fn = jax.vmap(
        lambda c_j: compute_conditional_expectation_beta(
            jnp.array([c_j]), confidences, labels, bandwidth
        ),
        in_axes=0,
    )
    cond_exps = cond_exp_fn(confidences)

    # Compute the p-norm calibration error
    bce = jnp.sum(jnp.abs(cond_exps - confidences) ** p)
    return (bce / N) ** (1 / p)


def test_nadaray(seed=0):
    key = random.PRNGKey(seed)
    num_samples = 5000
    num_classes = 5
    # Generate random probability distributions using Dirichlet
    key, subkey = random.split(key, 2)
    probs = random.dirichlet(subkey, alpha=jnp.ones(num_classes), shape=(num_samples,))
    # Sample true labels based on predicted probabilities
    labels = random.categorical(key, jnp.log(probs))
    key, subkey = random.split(key)
    labels_biased = random.categorical(subkey, 0.5 * jnp.log(probs))

    gamma = 0.1
    kernel = lambda x, y: jnp.exp(-((jnp.linalg.norm(x - y) / gamma) ** 2))

    # Compute Top-Class ECE
    cal = nadaraya_watson_ece(probs, labels, kernel)
    cal_biased = nadaraya_watson_ece(probs, labels_biased, kernel)

    print(f" ECE: {cal}")
    print(f" ECE (Biased): {cal_biased}")


def test_dirichlet(seed=0):
    key = random.PRNGKey(seed)
    num_samples = 5000
    num_classes = 5
    # Generate random probability distributions using Dirichlet
    key, subkey = random.split(key, 2)
    probs = random.dirichlet(subkey, alpha=jnp.ones(num_classes), shape=(num_samples,))
    # Sample true labels based on predicted probabilities
    labels = random.categorical(key, jnp.log(probs))
    key, subkey = random.split(key)
    labels_biased = random.categorical(subkey, 0.5 * jnp.log(probs))

    alphas = fit_dirichlet_alpha(probs)
    bandwidth = (jnp.mean(alphas) - jnp.mean(probs)) / 1
    print("Selected bandwidth", bandwidth)
    if bandwidth < 0:
        bandwidth = 0.001
    print("Adjusted bandwidth", bandwidth)

    # Compute Top-Class ECE
    dichlet_cal = dirichlet_calibration_error(probs, labels, bandwidth=bandwidth)
    dichlet_cal_biased = dirichlet_calibration_error(
        probs, labels_biased, bandwidth=bandwidth
    )

    print(f"Dirichlet ECE: {dichlet_cal}")
    print(f"Dirichlet ECE (Biased): {dichlet_cal_biased}")


def test_beta(seed=0):
    key = random.PRNGKey(seed)
    probs = random.uniform(key, (5000,))
    key1, key2 = random.split(key)
    labels = random.bernoulli(key1, probs).astype(jnp.float32)
    labels_biased = random.bernoulli(key2, probs * 0.8).astype(jnp.float32)

    ece = beta_calibration_error(probs, labels, 1)
    ece_biased = beta_calibration_error(probs, labels_biased, 1)
    print("Beta: {0} Beta Biased: {1}".format(ece, ece_biased))


if __name__ == "__main__":
    print("Dirichlet Calibration Error")
    test_dirichlet()
    print("Nadarray Watson with Gaussian Error")
    test_nadaray()
    print("Beta Calibration Error")
    test_beta()
