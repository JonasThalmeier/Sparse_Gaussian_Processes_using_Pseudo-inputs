from max_likelihood import SparseGPModel, run_likelihood_optimization, GP_predict
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

N, D, M = 50, 1, 10
X = jnp.array(np.random.randn(N, D))  # Convert to JAX array
# y =jnp.array(np.random.randn(N)) 
y = jnp.array(X + 0.2*X**2 + np.sin(X) + 0.1 * np.random.randn(N))  # Example target variable

# Extract results
X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt = run_likelihood_optimization(N, D, M, X, y)
X_pred, y_mean, y_std, samples = GP_predict(X,y)
# X_pred, y_mean, y_std, samples = GP_predict(X_bar_opt,y_)