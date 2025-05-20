from max_likelihood import SparseGPModel, run_likelihood_optimization, GP_predict
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Turn off interactive mode at the start
plt.ioff()

N, D, M = 70, 1, 20
key = jax.random.PRNGKey(41)
X = jax.random.uniform(key, shape=(N, D), minval=-0.5, maxval=1.5)
# Generate y with shape (N,) by using ravel()
y = jnp.array(jnp.mean(1.9*X+1.2*X**2-4.1*X**3+0.5*X**4+0.6*X**5+0.2*X**6 + np.sin(X), axis=1) + 0.3 * np.random.randn(N,))

# Extract results
model, f_bar, X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt, X_bar_init = run_likelihood_optimization(N, D, M, X, y)

X_pred_full, y_mean_full, y_std_full, samples_full = GP_predict(X, y, X.min()-0.5, X.max()+0.5,)
X_pred_sparse, y_mean_sparse, y_std_sparse, samples_sparse = GP_predict(
    X_bar_opt, 
    f_bar, 
    X.min()-0.5, 
    X.max()+0.5, 
    kernel=RBF(length_scale=jnp.exp(log_b_opt)) + WhiteKernel(noise_level=jnp.exp(log_sigma_sq_opt))
)

plt.figure(1)
plt.plot(X_pred_full, y_mean_full, 'b-', label='Mean prediction')
plt.fill_between(
    X_pred_full.ravel(),
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
plt.xlim(X_pred_full.min(), X_pred_full.max())

# Store the y-limits from first plot
y_min, y_max = plt.ylim()
x_min, x_max = plt.xlim()
plt.savefig('scatterplot_full.png', bbox_inches='tight', dpi=300)
plt.close()




plt.figure(2)
plt.plot(X_pred_sparse.ravel(), y_mean_sparse.ravel(), 'b-', label='Mean prediction')
plt.fill_between(
    X_pred_sparse.ravel(),  # Make sure X is 1D
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
