from sklearn.gaussian_process import GaussianProcessRegressor
from max_likelihood import SparseGPModel
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Turn off interactive mode at the start
plt.ioff()

N, D, M = 70, 1, 3
key = jax.random.PRNGKey(41)
X = jax.random.uniform(key, shape=(N, D), minval=-0.5, maxval=1.5)
# Generate y with shape (N,) by using ravel()
y = jnp.array(jnp.mean(1.9*X+1.2*X**2-4.1*X**3+0.5*X**4+0.6*X**5+0.2*X**6 + np.sin(X), axis=1) + 0.3 * np.random.randn(N,))

# Fit and predict with the full GP
X_pred = np.linspace(X.min()-0.5, X.max()+0.5, 500).reshape(-1, 1)
gp = GaussianProcessRegressor(
        kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3),
        alpha=1e-6,  # Add small noise to improve conditioning
        n_restarts_optimizer=5,
        normalize_y=True  # Normalize y values for better numerical stability
    )  
gp.fit(np.array(X), np.array(y))
y_mean_full, y_std_full = gp.predict(X_pred, return_std=True)

# Fit and predict with the sparse GP
model = SparseGPModel(N, D, M, margin=0.5)
f_bar, cov_f_bar, X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt, X_bar_init = model.fit(X, y)
y_mean_sparse, y_var_sparse = model.predict(X_pred)
y_std_sparse = jnp.sqrt(y_var_sparse)

plt.figure(1)
plt.plot(X_pred, y_mean_full, 'b-', label='Mean prediction')
plt.fill_between(
    X_pred.ravel(),
    y_mean_full.ravel() - 2 * y_std_full,
    y_mean_full.ravel() + 2 * y_std_full,
    color='blue',
    alpha=0.2,
    label='95% confidence interval'
)
plt.plot(X, y, 'ko', label='Observations')

# for i, sample in enumerate(samples.T):
#     plt.plot(X_pred, sample, lw=1, ls='--', alpha=0.7, label=f'Sample {i+1}' if i == 0 else None)

plt.legend()
plt.title("Gaussian Process Regression (Full Data)")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.xlim(X_pred.min(), X_pred.max())

# Store the y-limits from first plot
y_min, y_max = plt.ylim()
x_min, x_max = plt.xlim()
plt.savefig('scatterplot_full.png', bbox_inches='tight', dpi=300)
plt.close()




plt.figure(2)
plt.plot(X_pred.ravel(), y_mean_sparse.ravel(), 'b-', label='Mean prediction')
plt.fill_between(
    X_pred.ravel(),  # Make sure X is 1D
    (y_mean_sparse - 2 * y_std_sparse).ravel(),  # Make sure lower bound is 1D
    (y_mean_sparse + 2 * y_std_sparse).ravel(),  # Make sure upper bound is 1D
    color='blue',
    alpha=0.2,
    label='95% confidence interval'
)
plt.plot(X, y, 'ko', label='Observations')

# Add red "+" markers for inducing points (X_bar_opt)
# plt.plot(X_bar_opt, np.ones_like(X_bar_opt) * y_min + 0.25, 'r+', 
#          markersize=10, label='Inducing points', markeredgewidth=2)
plt.plot(X_bar_opt, f_bar, 'r+', 
         markersize=10, label='Inducing points', markeredgewidth=2)
plt.plot(X_bar_init, np.ones_like(X_bar_init) * y_max - 0.25, 'kx', 
         markersize=10, label='Initial inducing points', markeredgewidth=2)

plt.legend()
plt.title("Gaussian Process Regression (Sparse Data)")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True, linestyle='--', alpha=0.3)
# plt.tight_layout()
# Use the same y-limits as the first plot
plt.ylim(y_min, y_max)
plt.xlim(x_min, x_max)

plt.savefig('scatterplot_sparse.png', bbox_inches='tight', dpi=300)
plt.close()

# Show all plots and block until all windows are closed
