# A collection of calibration metric uses the idea of projection. 
import jax
import jax.numpy as jnp
from jax import scipy, random
from utils import sample_l_inf_ball_batch


def ece_equal_size_bins(
    probs: jnp.ndarray, labels: jnp.ndarray, num_bins: int = 10, p: float = 1.0
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equally sized probability bins for binary classification.

    Args:
        labels (jnp.ndarray): Binary ground truth labels (0 or 1) of shape (N,).
        probs (jnp.ndarray): Predicted probabilities for the positive class of shape (N,).
        num_bins (int): Number of equally sized bins.
        p (float): The norm to use for the calibration error (e.g., p=1 for absolute error, p=2 for squared error).

    Returns:
        float: The computed ECE value.
    """
    # Ensure inputs are JAX arrays

    probs = jnp.asarray(probs)
    labels = jnp.asarray(labels)
    if len(probs.shape) != 1 or probs.shape != labels.shape:
        raise ValueError("Wrong shapes {0}, {1}".format(probs.shape, labels.shape))

    # Define bin edges
    bin_edges = jnp.linspace(0.0, 1.0, num_bins + 1)

    # Assign each probability to a bin
    bin_indices = jnp.digitize(probs, bin_edges, right=False) - 1
    bin_indices = jnp.clip(
        bin_indices, 0, num_bins - 1
    )  # Handle edge case where prob == 1.0

    # Compute bin totals using bincountv
    bin_totals = jnp.bincount(bin_indices, minlength=num_bins).astype(jnp.float32)

    # Compute bin correct predictions using bincount
    bin_labels = jnp.bincount(bin_indices, weights=labels, minlength=num_bins).astype(
        jnp.float32
    )

    # Compute bin confidence sums using bincount
    bin_confs = jnp.bincount(bin_indices, weights=probs, minlength=num_bins).astype(
        jnp.float32
    )

    # Compute accuracy and confidence for each bin (avoid division by zero)
    bin_accuracy = bin_labels / (bin_totals + 1e-10)
    bin_confidence = bin_confs / (bin_totals + 1e-10)

    # Compute the p-norm ECE
    ece = jnp.sum(jnp.abs(bin_accuracy - bin_confidence) ** p * bin_totals) / jnp.sum(
        bin_totals
    )
    return ece ** (1 / p)


def ece_equal_weight_bins(
    probs: jnp.ndarray, labels: jnp.ndarray, num_bins: int = 10, p: float = 1.0
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equally weighted bins for binary classification.
    This implementation uses quantile-based bin edges and digitizes the probabilities accordingly.

    Args:
        labels (jnp.ndarray): Binary ground truth labels (0 or 1) of shape (N,).
        probs (jnp.ndarray): Predicted probabilities for the positive class of shape (N,).
        num_bins (int): Number of equally weighted bins.
        p (float): The norm to use for the calibration error (e.g., p=1 for absolute error, p=2 for squared error).

    Returns:
        float: The computed ECE value.
    """
    # Compute quantile-based bin edges
    quantiles = jnp.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = jnp.quantile(probs, quantiles)

    if len(probs.shape) != 1 or probs.shape != labels.shape:
        raise ValueError("Wrong shapes {0}, {1}".format(probs.shape, labels.shape))

    # To handle possible duplicate edges, slightly adjust the bin_edges
    # Ensure that the first edge is 0.0 and the last edge is 1.0
    bin_edges = jnp.concatenate((jnp.array([0.0]), bin_edges[1:-1], jnp.array([1.0])))

    # Assign each probability to a bin
    bin_indices = jnp.digitize(probs, bin_edges, right=True) - 1
    bin_indices = jnp.clip(
        bin_indices, 0, num_bins - 1
    )  # Ensure bin indices are within [0, num_bins-1]

    # Compute bin totals, correct predictions, and confidence sums using bincount
    bin_totals = jnp.bincount(bin_indices, minlength=num_bins).astype(jnp.float32)
    bin_labels = jnp.bincount(bin_indices, weights=labels, minlength=num_bins).astype(
        jnp.float32
    )
    bin_confs = jnp.bincount(bin_indices, weights=probs, minlength=num_bins).astype(
        jnp.float32
    )

    # Compute accuracy and confidence for each bin (avoid division by zero)
    bin_accuracy = bin_labels / (bin_totals + 1e-10)
    bin_confidence = bin_confs / (bin_totals + 1e-10)

    # Compute the p-norm ECE
    ece = jnp.sum(jnp.abs(bin_accuracy - bin_confidence) ** p * bin_totals) / jnp.sum(
        bin_totals
    )
    return ece ** (1 / p)


def top_calibration_error(
    probs: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
    p: float = 1.0,
    equal_size: bool = True,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for multi-class classification based on top-class.

    Args:
        probs (jnp.ndarray): Predicted probabilities for each class of shape (N, C).
        labels (jnp.ndarray): True class labels of shape (N,).
        num_bins (int): Number of bins.
        p (float): Norm for calibration error.
        equal_size (bool): If True, use equal size bins; else use equal weight bins.

    Returns:
        float: Computed ECE.
    """
    # Compute top-class probabilities and correctness
    top_probs = jnp.max(probs, axis=1)
    predictions = jnp.argmax(probs, axis=1)
    correct = (predictions == labels).astype(jnp.float32)
    if equal_size:
        ece = ece_equal_size_bins(top_probs, correct, num_bins=num_bins, p=p)
    else:
        ece = ece_equal_weight_bins(top_probs, correct, num_bins=num_bins, p=p)
    return ece


def class_wise_calibration_error(
    probs: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
    p: float = 1.0,
    equal_size: bool = True,
) -> jnp.ndarray:
    """
    Compute Class-Wise Expected Calibration Error (ECE) for multi-class classification.

    Args:
        probs (jnp.ndarray): Predicted probabilities for each class of shape (N, C).
        labels (jnp.ndarray): True class labels of shape (N,).
        num_bins (int): Number of bins.
        p (float): Norm for calibration error.
        equal_size (bool): If True, use equal size bins; else use equal weight bins.

    Returns:
        jnp.ndarray: Array of ECE values for each class of shape (C,).
    """
    num_classes = probs.shape[1]
    one_hot_labels = jax.nn.one_hot(labels, num_classes).astype(jnp.float32)
    class_ece = []

    for c in range(num_classes):
        class_probs = probs[:, c]
        class_correct = one_hot_labels[:, c]

        if equal_size:
            ece = ece_equal_size_bins(
                class_probs, class_correct, num_bins=num_bins, p=p
            )
        else:
            ece = ece_equal_weight_bins(
                class_probs, class_correct, num_bins=num_bins, p=p
            )

        class_ece.append(ece)

    return jnp.mean(jnp.array(class_ece))


def projected_calibration_error(
    key,
    probs: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
    p: float = 1.0,
    equal_size: bool = True,
    num_proj=100,
) -> jnp.ndarray:
    """Compute the calibration error by first projecting onto a random direction from the l_infty ball
    then measuring the 1D calibration error along the projected direction. This can be seen as a generalization
    of class-wise calibration.
    """
    key, _ = random.split(key, 2)
    probs = jnp.array(probs)
    labels = jnp.array(labels)
    num_classes = probs.shape[1]
    proj_vecs = jnp.abs(sample_l_inf_ball_batch(key, num_proj, num_classes))

    proj_probs = probs @ proj_vecs.T  # output is num_samples x num_proj
    # proj_probs  = (proj_probs+1)/2 #insuring it is still in zero one
    one_hot_labels = jax.nn.one_hot(labels, num_classes).astype(jnp.float32)
    proj_labels = one_hot_labels @ proj_vecs.T
    # proj_labels = (one_hot_labels+1)/2

    class_ece = []
    for c in range(num_classes):
        class_probs = proj_probs[:, c]
        class_correct = proj_labels[:, c]

        if equal_size:
            ece = ece_equal_size_bins(
                class_probs, class_correct, num_bins=num_bins, p=p
            )
        else:
            ece = ece_equal_weight_bins(
                class_probs, class_correct, num_bins=num_bins, p=p
            )

        class_ece.append(ece)

    return jnp.mean(jnp.array(class_ece))


def test_ece_1d_random(num_bins=20, seed=0):
    key = random.PRNGKey(seed)
    probs = random.uniform(key, (5000,))
    key1, key2 = random.split(key)
    labels = random.bernoulli(key1, probs).astype(jnp.float32)
    labels_biased = random.bernoulli(key2, probs * 0.8).astype(jnp.float32)
    ece_size = ece_equal_size_bins(probs, labels, num_bins=num_bins, p=1.0)
    ece_weight = ece_equal_weight_bins(probs, labels, num_bins=num_bins, p=1.0)
    ece_size_biased = ece_equal_size_bins(
        probs, labels_biased, num_bins=num_bins, p=1.0
    )
    ece_weight_biased = ece_equal_weight_bins(
        probs, labels_biased, num_bins=num_bins, p=1.0
    )
    print(
        "Random ECE tests for perfect calibration: size: {0} weight: {1}".format(
            ece_size, ece_weight
        )
    )
    print(
        "Random ECE tests for biased : size: {0} weight: {1}".format(
            ece_size_biased, ece_weight_biased
        )
    )


def test_ece_multiclass_random(num_bins=20, seed=0, equal_size=True):
    key = random.PRNGKey(seed)
    num_samples = 5000
    num_classes = 5

    # Generate random probability distributions using Dirichlet
    key, subkey, proj_key = random.split(key, 3)
    probs = random.dirichlet(subkey, alpha=jnp.ones(num_classes), shape=(num_samples,))

    # Sample true labels based on predicted probabilities
    labels = random.categorical(key, jnp.log(probs))

    key, subkey = random.split(key)
    labels_biased = random.categorical(subkey, 0.5 * jnp.log(probs))

    # Compute Top-Class ECE
    ece_top = top_calibration_error(
        probs, labels, num_bins=num_bins, p=1.0, equal_size=equal_size
    )
    ece_top_biased = top_calibration_error(
        probs, labels_biased, num_bins=num_bins, p=1.0, equal_size=equal_size
    )

    # Compute Class-Wise ECE
    ece_class = class_wise_calibration_error(
        probs, labels, num_bins=num_bins, p=1.0, equal_size=equal_size
    )
    ece_class_biased = class_wise_calibration_error(
        probs, labels_biased, num_bins=num_bins, p=1.0, equal_size=equal_size
    )

    # Compute Projected Calibration Error
    proj_class = projected_calibration_error(
        proj_key,
        probs,
        labels,
        num_bins=num_bins * 2,
        num_proj=100,
        p=1.0,
        equal_size=equal_size,
    )
    proj_class_biased = projected_calibration_error(
        proj_key,
        probs,
        labels_biased,
        num_bins=num_bins * 2,
        num_proj=100,
        p=1.0,
        equal_size=equal_size,
    )

    print(f"Top-Class ECE: {ece_top}")
    print(f"Top-Class ECE (Biased): {ece_top_biased}")
    print(f"Class-Wise ECE: {ece_class}")
    print(f"Class-Wise ECE (Biased): {ece_class_biased}")
    print(f"Projected ECE: {proj_class}")
    print(f"Projected ECE (Biased): {proj_class_biased}")


if __name__ == "__main__":
    print("test_ece_1d_random")
    test_ece_1d_random()
    print("multiclass equal size bins")
    test_ece_multiclass_random(equal_size=False)
    print("multiclass equal weight bins ")
    test_ece_multiclass_random(equal_size=True)
