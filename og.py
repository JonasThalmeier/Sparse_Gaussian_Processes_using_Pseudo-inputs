import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import value_and_grad
from scipy.optimize import minimize
import numpy as np

def kernel(X1, X2, log_c, log_b):
    c = jnp.exp(log_c)
    b = jnp.exp(log_b)
    diff = X1[:, None, :] - X2[None, :, :]
    sq_dist = jnp.sum(b * (diff ** 2), axis=2)
    return c * jnp.exp(-0.5 * sq_dist)

def pack_params(X_bar, log_c, log_b, log_sigma_sq):
    return jnp.concatenate([X_bar.ravel(), log_c[None], log_b, log_sigma_sq[None]])

def unpack_params(params, M, D):
    X_bar = params[:M*D].reshape(M, D)
    log_c = params[M*D]
    log_b = params[M*D + 1 : M*D + 1 + D]
    log_sigma_sq = params[M*D + 1 + D]
    return X_bar, log_c, log_b, log_sigma_sq

@jax.jit
def neg_log_likelihood(params, X, y, M, D, jitter=1e-6):
    X_bar, log_c, log_b, log_sigma_sq = unpack_params(params, M, D)
    c = jnp.exp(log_c)
    b = jnp.exp(log_b)
    sigma_sq = jnp.exp(log_sigma_sq)
    
    # Compute kernel matrices with jitter
    K_M = kernel(X_bar, X_bar, log_c, log_b) + jitter * jnp.eye(M)
    L = jax.scipy.linalg.cholesky(K_M, lower=True)
    
    K_NM = kernel(X, X_bar, log_c, log_b)
    U_T = jax.scipy.linalg.solve_triangular(L, K_NM.T, lower=True)
    U = U_T.T
    
    d = c + sigma_sq
    y = y.reshape(-1, 1)
    term1 = (y.T @ y) / d
    
    term2_part1 = (U.T @ y) / d
    S = (U.T @ U) / d + jnp.eye(M) + jitter * jnp.eye(M)
    S_inv = jax.scipy.linalg.inv(S)
    term2 = term2_part1.T @ S_inv @ term2_part1
    
    quad_form = term1 - term2
    log_det_D = X.shape[0] * jnp.log(d)
    L_S = jax.scipy.linalg.cholesky(S, lower=True)
    log_det_S = 2 * jnp.sum(jnp.log(jnp.diag(L_S)))
    log_det = log_det_D + log_det_S
    
    nll = 0.5 * (quad_form + log_det + X.shape[0] * jnp.log(2 * jnp.pi))
    return nll.squeeze()

# Example setup
N, D, M = 100, 2, 20
np.random.seed(0)
X = np.random.randn(N, D)
y = np.random.randn(N)

# Initialize parameters
X_bar_init = X[np.random.choice(N, M, replace=False)]
log_c_init = np.log(1.0)
log_b_init = np.zeros(D)
log_sigma_sq_init = np.log(0.1)
params_init = pack_params(X_bar_init, log_c_init, log_b_init, log_sigma_sq_init)

# Optimize
obj_grad = jax.jit(value_and_grad(neg_log_likelihood))
result = minimize(lambda p: obj_grad(p, X, y, M, D),
                 params_init,
                 method='L-BFGS-B',
                 jac=True)

# Extract results
X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt = unpack_params(result.x, M, D)
c_opt = np.exp(log_c_opt)
b_opt = np.exp(log_b_opt)
sigma_sq_opt = np.exp(log_sigma_sq_opt)

print("Optimized pseudo-inputs:\n", X_bar_opt)
print("Optimized c:", c_opt)
print("Optimized b:", b_opt)
print("Optimized sigma²:", sigma_sq_opt)
