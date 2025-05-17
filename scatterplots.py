from max_likelihood import SparseGPModel, run_likelihood_optimization, GP_predict
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

# Turn off interactive mode at the start
plt.ioff()

N, D, M = 50, 1, 10
X = jnp.array(np.random.randn(N, D))  # Convert to JAX array
# Generate y with shape (N,) by using ravel()
y = jnp.array(jnp.mean(X + 0.2*X**2 + np.sin(X), axis=1) + 0.1 * np.random.randn(N,))

# Extract results
f_bar, X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt = run_likelihood_optimization(N, D, M, X, y)

# First plot
plt.figure(1)
X_pred, y_mean, y_std, samples = GP_predict(X, y)

# Second plot
plt.figure(2)
X_pred, y_mean, y_std, samples = GP_predict(X_bar_opt, f_bar)

# Show all plots and block until all windows are closed
plt.show()