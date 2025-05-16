import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import value_and_grad
from scipy.optimize import minimize
import numpy as np
from jax.scipy.stats import multivariate_normal
from functools import partial

def kernel(X1, X2, log_c, log_b):
    c = jnp.exp(log_c)
    b = jnp.exp(log_b)
    diff = X1[:, None, :] - X2[None, :, :]
    sq_dist = jnp.sum(b * (diff ** 2), axis=2)
    return c * jnp.exp(-0.5 * sq_dist)

def pack_params(X_bar, log_c, log_b, log_sigma_sq):
    return jnp.concatenate([X_bar.ravel(), log_c[None], log_b, log_sigma_sq[None]])

@partial(jax.jit, static_argnums=(1, 2))
def unpack_params(params, M, D):
    """Unpack parameters with static shapes"""
    # Use static indexing by making M and D static arguments
    X_bar = params[:M*D].reshape(M, D)
    log_c = params[M*D]
    log_b = params[M*D + 1:M*D + 1 + D]
    log_sigma_sq = params[M*D + 1 + D]
    
    return X_bar, log_c, log_b, log_sigma_sq

@jax.jit
def neg_log_likelihood(params, M, D, X, y, jitter=1e-6):
    X_bar, log_c, log_b, log_sigma_sq = unpack_params(params, M, D)
    # log_likelihood = multivariate_normal.logpdf(y, mean=0.0, cov=K)
    c = jnp.exp(log_c)
    b = jnp.exp(log_b)
    sigma_sq = jnp.exp(log_sigma_sq)
    
    # Ensure y is properly shaped (N, 1)
    y = y.reshape(-1, 1)
    
    # Compute kernel matrices with jitter
    K_M = kernel(X_bar, X_bar, log_c, log_b) + jitter * jnp.eye(M)    
    K_NM = kernel(X, X_bar, log_c, log_b)
    K_M_inv = jnp.linalg.inv(K_M)
    quad_terms = jnp.sum(K_NM @ K_M_inv * K_NM.T, axis=1)

    # K_nn is just the diagonal kernel value: c
    lambda_diag = c - quad_terms
    
    nll = -multivariate_normal.logpdf(y, mean=0.0, cov=quad_terms + jnp.diag(lambda_diag) + sigma_sq * jnp.eye()+ jitter)
    return nll.squeeze()

# Example setup
N, D, M = 100, 2, 20
np.random.seed(0)
X = jnp.array(np.random.randn(N, D))  # Convert to JAX array
y = jnp.array(np.random.randn(N))     # Convert to JAX array

# Initialize parameters
X_bar_init = jnp.array(X[jax.random.choice(jax.random.PRNGKey(0), N, shape=(M,), replace=False)])
log_c_init = jnp.log(1.0)
log_b_init = jnp.zeros(D)
log_sigma_sq_init = jnp.log(0.1)
params_init = pack_params(X_bar_init, log_c_init, log_b_init, log_sigma_sq_init)

# Optimize
obj_grad = jax.value_and_grad(lambda p, M, D, X, y: neg_log_likelihood(p, M, D, X, y))
result = minimize(
    lambda p: obj_grad(p, M, D, X, y)[0],
    params_init,
    method='L-BFGS-B',
    jac=lambda p: obj_grad(p, M, D, X, y)[1]
)

# Extract results
X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt = unpack_params(result.x, M, D)
c_opt = jnp.exp(log_c_opt)
b_opt = jnp.exp(log_b_opt)
sigma_sq_opt = jnp.exp(log_sigma_sq_opt)

print("Optimized pseudo-inputs:\n", X_bar_opt)
print("Optimized c:", c_opt)
print("Optimized b:", b_opt)
print("Optimized sigma²:", sigma_sq_opt)
