from sklearn.gaussian_process import GaussianProcessRegressor
from max_likelihood import SparseGPModel
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Turn off interactive mode at the start
plt.ioff()

# Generate synthetic data
N, D, M = 150, 1, 5
key = jax.random.PRNGKey(41)
X = jax.random.uniform(key, shape=(N, D), minval=-0.5, maxval=1.5)
y = jnp.array(jnp.mean(1.9*X+1.2*X**2-4.1*X**3+0.5*X**4+0.6*X**5+0.2*X**6 + np.sin(X), axis=1) + 0.333 * np.random.randn(N,))


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
model = SparseGPModel(N, D, M, margin=0)
f_bar, cov_f_bar, X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt, X_bar_init = model.fit(X, y)
y_mean_sparse, y_var_sparse = model.predict(X_pred)
y_std_sparse = jnp.sqrt(y_var_sparse)

# Define consistent plotting parameters
fig_size = (10, 6)
font_size = 12
plt.style.use('default')  # Reset to default style
plt.rc('font', size=font_size)
plt.rc('axes', titlesize=font_size+2, labelsize=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.rc('legend', fontsize=font_size)

def create_plot(x_pred, y_mean, y_std, X, y, title, inducing_points=None, f_bar=None, X_init=None):
    fig = plt.figure(figsize=fig_size, dpi=300)
    plt.plot(x_pred, y_mean, 'b-', label='Mean prediction', linewidth=2)
    plt.fill_between(
        x_pred.ravel(),
        (y_mean - 2 * y_std).ravel(),
        (y_mean + 2 * y_std).ravel(),
        color='blue',
        alpha=0.2,
        label='95% confidence interval'
    )
    plt.plot(X, y, 'ko', label='Observations', markersize=4)
    
    if inducing_points is not None:
        plt.plot(inducing_points, -np.ones_like(X_init), 'r+', 
                markersize=10, label='Inducing points', markeredgewidth=2)
    if X_init is not None:
        plt.plot(X_init, np.ones_like(X_init)*3, 'kx', 
                markersize=10, label='Initial inducing points', markeredgewidth=2)
    
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

# First plot (Full GP)
fig1 = create_plot(X_pred, y_mean_full, y_std_full, X, y, 
                  "Gaussian Process Regression (Full Data)")
y_min, y_max = plt.ylim()
x_min, x_max = plt.xlim()
plt.savefig('scatterplot_full.png', bbox_inches='tight', dpi=300)
plt.close()

# Second plot (Sparse GP)
fig2 = create_plot(X_pred, y_mean_sparse, y_std_sparse, X, y,
                  "Gaussian Process Regression (Sparse Data)",
                  inducing_points=X_bar_opt, f_bar=f_bar, X_init=X_bar_init)
plt.ylim(y_min, y_max)
plt.xlim(x_min, x_max)
plt.savefig('scatterplot_sparse.png', bbox_inches='tight', dpi=300)
plt.close()

# Show all plots and block until all windows are closed
