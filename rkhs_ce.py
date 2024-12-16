import jax.numpy as jnp
import jax 

def kernel_calibration_error_1d(residuals, gram_mat):
    # return the kernel calibration error of norm 1.
    coef = residuals.reshape(-1, 1)
    squared_rkhs_norm = jnp.dot(jnp.dot(coef.T, gram_mat), coef)
    return jnp.sqrt(squared_rkhs_norm)

vec_kernel_calibration_error = jax.vmap(kernel_calibration_error_1d,in_axes=(1,None), out_axes=1)

def kernel_calibration_error(probs, labels, kernel_fun):
    N, K = probs.shape
    onehot_labels = jax.nn.one_hot(labels, num_classes=K)
    residuals  = probs-onehot_labels
    gram_matrix = kernel_fun(probs, probs)
    dim_error = vec_kernel_calibration_error(residuals, gram_matrix)
    return jnp.sum(dim_error)
