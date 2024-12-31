
import jax 
import jax.numpy as jnp


def generative_model(key, alpha, beta, pi, n_samples):
    """
    Simulates the generative model described in the text.

    Parameters:
    - key: JAX random key.
    - alpha: Dirichlet concentration parameters (m-dimensional).
    - beta: Category probabilities for Y|Z=0 (m-dimensional).
    - pi: Bernoulli parameter for Z.
    - n_samples: Number of samples to generate.

    Returns:
    - g_X: Predictions g(X), sampled from Dir(alpha).
    - Z: Latent variable Z, sampled from Ber(pi).
    - Y: Labels Y, sampled conditionally on Z and g(X).
    """
    keys = jax.random.split(key, 3)

    # Sample g(X) ~ Dir(alpha)
    g_X = jax.random.dirichlet(keys[0], alpha, shape=(n_samples,))

    # Sample Z ~ Ber(pi)
    Z = jax.random.bernoulli(keys[1], p=pi, shape=(n_samples,)).astype(jnp.float32)

    # Compute the combined distribution for Y
    dist = (1 - Z[:, None]) * g_X + Z[:, None] * beta[None, :]

    # Sample Y from the categorical distribution using dist
    Y = jax.random.categorical(keys[2], logits=jnp.log(dist), axis=1)

    return g_X, Z, Y
