import jax.numpy as jnp
import jax 

def kernel_calibration_error_1d(residuals, gram_mat):
    # return the kernel calibration error of norm 1.
    coef = residuals.reshape(-1, 1)
    squared_rkhs_norm = jnp.dot(jnp.dot(coef.T, gram_mat), coef)
    return jnp.squeeze(jnp.sqrt(squared_rkhs_norm))

vec_kernel_calibration_error = jax.vmap(kernel_calibration_error_1d,in_axes=(1,None), out_axes=0)

def kernel_calibration_error(probs, labels, kernel_fun):
    N, K = probs.shape
    onehot_labels = jax.nn.one_hot(labels, num_classes=K)
    residuals  = probs-onehot_labels
    gram_matrix = kernel_fun(probs, probs)
    dim_error = vec_kernel_calibration_error(residuals, gram_matrix)
    return jnp.sum(dim_error)/N


def biased_kernel_cal_error(probs, labels, kernel_fun):
    N, K = probs.shape
    onehot_labels = jax.nn.one_hot(labels, num_classes=K)
    residuals  = probs-onehot_labels
    gram_matrix = kernel_fun(probs, probs) 
    residuals_outer = residuals@residuals.T
    err_mat = gram_matrix*residuals_outer
    return jnp.mean(err_mat)



def debiased_kernel_cal_error(probs, labels, kernel_fun):
    N, K = probs.shape
    onehot_labels = jax.nn.one_hot(labels, num_classes=K)
    residuals  = probs-onehot_labels
    gram_matrix = kernel_fun(probs, probs) 
    residuals_outer = residuals@residuals.T
    err_mat = gram_matrix*residuals_outer
    rows, cols = jnp.tril_indices(n=err_mat.shape[0], k=-1)
    below_diag = err_mat[rows, cols]
    return jnp.mean(below_diag)


if __name__=="__main__":
    K = 10
    N = 1000
    seed = jax.random.PRNGKey(42)
    from synth_data import generative_model
    key1, key2 = jax.random.split(seed,2)
    good_probs,_,good_Y = generative_model(key1,jnp.ones(K,)*(1/K), jnp.ones(K,)*(1/K),0.0,N)
    bad_probs,_,bad_Y = generative_model(key1,jnp.ones(K,)*(1/K), jnp.ones(K,)*(1/K),0.5,N)

    
    def get_kernel_fun(sigma):
        kern_fun = lambda x,y : jnp.exp(-(jnp.linalg.norm(x-y)/sigma)**2)
        kern_fun = jax.vmap(kern_fun, (0,None),0)
        kern_fun = jax.vmap(kern_fun,(None,0),1)
        return kern_fun
    
    
    
    for sigma in [0.01, 0.1, 0.2, 0.5]:
        kern_fun = get_kernel_fun(sigma)
        print("Good data")
        print("Original", kernel_calibration_error(good_probs,good_Y,kern_fun))
        print("Biased", biased_kernel_cal_error(good_probs,good_Y,kern_fun))
        print("Debiased", debiased_kernel_cal_error(good_probs,good_Y,kern_fun))
        print("Bad data")
        print("Original", kernel_calibration_error(bad_probs,bad_Y,kern_fun))
        print("Biased", biased_kernel_cal_error(bad_probs,bad_Y,kern_fun))
        print("Debiased", debiased_kernel_cal_error(bad_probs,bad_Y,kern_fun))
