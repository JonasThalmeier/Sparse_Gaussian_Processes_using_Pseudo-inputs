import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import value_and_grad
from scipy.optimize import minimize
from jax.scipy.stats import multivariate_normal
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
import numpy as np

class SparseGPModel:
    def __init__(self, N, D, M):
        self.N = N
        self.D = D
        self.M = M
        
    @partial(jax.jit, static_argnums=0)
    def kernel(self, X1, X2, log_c, log_b):
        c = jnp.exp(log_c)
        b = jnp.exp(log_b)
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = jnp.sum(b * (diff ** 2), axis=2)
        return c * jnp.exp(-0.5 * sq_dist)
    
    def pack_params(self, X_bar, log_c, log_b, log_sigma_sq):
        return jnp.concatenate([X_bar.ravel(), log_c[None], log_b, log_sigma_sq[None]])
    
    @partial(jax.jit, static_argnums=0)
    def unpack_params(self, params):
        X_bar = params[:self.M*self.D].reshape(self.M, self.D)
        log_c = params[self.M*self.D]
        log_b = params[self.M*self.D + 1:self.M*self.D + 1 + self.D]
        log_sigma_sq = params[self.M*self.D + 1 + self.D]
        return X_bar, log_c, log_b, log_sigma_sq
    
    @partial(jax.jit, static_argnums=0)
    def neg_log_likelihood(self, params, X, y, jitter=1e-6):
        X_bar, log_c, log_b, log_sigma_sq = self.unpack_params(params)
        c = jnp.exp(log_c)
        b = jnp.exp(log_b)
        sigma_sq = jnp.exp(log_sigma_sq)
        
        K_M = self.kernel(X_bar, X_bar, log_c, log_b) + jitter * jnp.eye(self.M)    
        K_NM = self.kernel(X, X_bar, log_c, log_b)
        K_M_inv = jnp.linalg.inv(K_M)
        quad_terms = jnp.sum(K_NM @ K_M_inv @ K_NM.T, axis=1)
        
        # Reshape y to (N,1) for kernel computation
        y_reshaped = y.reshape(-1, 1)
        K_y = self.kernel(y_reshaped, y_reshaped, log_c, log_b)
        lambda_diag = jnp.diag(K_y) - quad_terms
        
        # Construct the covariance matrix
        cov = quad_terms[:, None] + jnp.diag(lambda_diag) + sigma_sq * jnp.eye(self.N) + jitter * jnp.eye(self.N)
        mean = jnp.zeros_like(y)
        
        nll = -jnp.sum(multivariate_normal.logpdf(y, mean=mean, cov=cov))
        return nll
    
    def mean_f_bar(self, X, y, X_bar, log_c, log_b, log_sigma_sq, jitter=1e-6):
        c = jnp.exp(log_c)
        b = jnp.exp(log_b)
        sigma_sq = jnp.exp(log_sigma_sq)
        
        K_M = self.kernel(X_bar, X_bar, log_c, log_b) + jitter * jnp.eye(self.M)    
        K_NM = self.kernel(X, X_bar, log_c, log_b)
        K_M_inv = jnp.linalg.inv(K_M)
        quad_terms = jnp.sum(K_NM @ K_M_inv @ K_NM.T, axis=1)
        
        y_reshaped = y.reshape(-1, 1)
        K_y = self.kernel(y_reshaped, y_reshaped, log_c, log_b)
        lambda_diag = jnp.diag(K_y) - quad_terms
        Q = K_M + K_NM.T @ (jnp.diag(lambda_diag) + sigma_sq * jnp.eye(self.N)) @ K_NM
        Q_inv = jnp.linalg.inv(Q)
        mean = K_M @ Q_inv @ K_NM.T @ (jnp.diag(lambda_diag) + sigma_sq * jnp.eye(self.N)) @ y
        return mean

def run_likelihood_optimization(N, D, M, X, y):
    model = SparseGPModel(N, D, M)
    
    # Enable float64 support in JAX
    jax.config.update("jax_enable_x64", True)
    
    # Initialize parameters
    X_bar_init = jnp.array(X[jax.random.choice(jax.random.PRNGKey(0), N, shape=(M,), replace=False)])
    log_c_init = jnp.array(0.0)  # log(1.0)
    log_b_init = jnp.zeros(D)
    log_sigma_sq_init = jnp.array(-2.3)  # log(0.1)
    params_init = model.pack_params(X_bar_init, log_c_init, log_b_init, log_sigma_sq_init)

    # Define objective function outside of class context
    def neg_log_likelihood_fn(params):
        return model.neg_log_likelihood(params, X, y)

    # Create value and gradient function
    obj_grad = jax.value_and_grad(neg_log_likelihood_fn)

    # Define objective for scipy optimizer
    def objective(params_np):
        params_jax = jnp.array(params_np)
        value, grad = obj_grad(params_jax)
        return np.array(value), np.array(grad)

    # Convert to NumPy array for scipy optimizer
    params_init_np = np.array(params_init)
    
    result = minimize(
        lambda p: objective(p)[0],
        params_init_np,
        method='L-BFGS-B',
        jac=lambda p: objective(p)[1]
    )
    X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt = model.unpack_params(jnp.array(result.x))
    f_bar = model.mean_f_bar(X, y, X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt)
    return f_bar, X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt

def GP_predict(X, y,
               kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2),
               plot=True):
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)


    # Fit GP
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    gp.fit(X, y)

    # Test points for prediction
    X_pred = np.linspace(X.min() - 1, X.max() + 1, 1000).reshape(-1, 1)
    y_mean, y_std = gp.predict(X_pred, return_std=True)

    # Draw samples from posterior
    samples = gp.sample_y(X_pred, n_samples=3, random_state=0)

    if plot:
        # Plot
        # plt.figure(figsize=(10, 5))
        plt.plot(X_pred, y_mean, 'b-', label='Mean prediction')
        plt.fill_between(
            X_pred.ravel(),
            y_mean.ravel() - 2 * y_std,
            y_mean.ravel() + 2 * y_std,
            color='blue',
            alpha=0.2,
            label='95% confidence interval'
        )
        plt.plot(X, y, 'ko', label='Observations')

        for i, sample in enumerate(samples.T):
            plt.plot(X_pred, sample, lw=1, ls='--', alpha=0.7, label=f'Sample {i+1}' if i == 0 else None)

        plt.legend()
        plt.title("Gaussian Process Regression")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)

    return X_pred, y_mean, y_std, samples


# # Example usage
# N, D, M = 100, 2, 20
# key = jax.random.PRNGKey(0)
# X = jax.random.normal(key, shape=(N, D))
# y = jax.random.normal(key, shape=(N,))

# # Extract results
# X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt = run_likelihood_optimization(N, D, M, X, y)

# c_opt = jnp.exp(log_c_opt)
# b_opt = jnp.exp(log_b_opt)
# sigma_sq_opt = jnp.exp(log_sigma_sq_opt)

# print("Optimized pseudo-inputs:\n", X_bar_opt)
# print("Optimized c:", c_opt)
# print("Optimized b:", b_opt)
# print("Optimized sigma²:", sigma_sq_opt)
