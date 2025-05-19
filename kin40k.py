from max_likelihood import SparseGPModel, run_likelihood_optimization, GP_predict
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from uci_datasets import Dataset
import time
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Load data
data = Dataset("kin40k")
x_train, y_train, x_test, y_test = data.get_split(split=0)

# Define number of inducing points to test
# inducing_points = [10, 20, 50, 100, 200, 500]
inducing_points = np.linspace(50, 1500, num=15, dtype=int)  # Logarithmic scale for inducing points
# N = len(x_train)  # Use full training set
N = 1000
D = x_train.shape[1]

# Initialize arrays to store results
mse_scores = []
train_times = []
inference_times = []

# Ensure consistent sizes between training and test sets
N_train = 5000  # Size of training set
N_test = 1000   # Size of test set

# Slice the data to use consistent sizes
X = jnp.array(x_train[:N_train])
y = jnp.array(y_train[:N_train]).reshape(-1)
X_test = jnp.array(x_test[:N_test])
y_test = jnp.array(y_test[:N_test]).reshape(-1)

for M in inducing_points:
    print(f"Training with {M} inducing points...")
    
    # Measure training time
    start_time = time.time()
    f_bar, X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt = run_likelihood_optimization(N_train, D, M, X, y)
    train_time = time.time() - start_time
    train_times.append(train_time)
    
    # Measure inference time
    start_time = time.time()
    model = SparseGPModel(N_test, D, M)
    # f_bar_test = model.mean_f_bar(X_test, y_test, X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt)
    kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    gp.fit(X_bar_opt, f_bar)
    f_bar_test,_ = gp.predict(X_test, return_std=True)
    
    inference_time = time.time() - start_time
    inference_times.append(inference_time)
    
    # Calculate MSE
    mse = mean_squared_error(y_test, f_bar_test)
    mse_scores.append(mse)

# Create and save MSE plot with logarithmic y-axis
plt.figure(figsize=(8, 6))
plt.semilogy(inducing_points, mse_scores, 'k-o', color='black')  # Use semilogy for log scale on y-axis
plt.xlabel('Number of Inducing Points (M)')
plt.ylabel('Mean Squared Error (log scale)')
plt.title('MSE vs Number of Inducing Points')
plt.grid(True, which="both", ls="-")  # Add grid lines for both major and minor ticks
plt.tight_layout()
plt.savefig('mse_plot.png', bbox_inches='tight', dpi=300)
plt.close()

# Create and save training time plot
plt.figure(figsize=(8, 6))
plt.plot(inducing_points, train_times, 'k-o', color='black')
plt.xlabel('Number of Inducing Points (M)')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time vs Number of Inducing Points')
plt.grid(True)
plt.tight_layout()
plt.savefig('training_time_plot.png', bbox_inches='tight', dpi=300)
plt.close()

# Create and save inference time plot
plt.figure(figsize=(8, 6))
plt.plot(inducing_points, inference_times, 'k-o', color='black')
plt.xlabel('Number of Inducing Points (M)')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time vs Number of Inducing Points')
plt.grid(True)
plt.tight_layout()
plt.savefig('inference_time_plot.png', bbox_inches='tight', dpi=300)
plt.close()

# Print numerical results
print("\nNumerical Results:")
print("Inducing Points | MSE | Training Time | Inference Time")
print("-" * 55)
for m, mse, t_time, i_time in zip(inducing_points, mse_scores, train_times, inference_times):
    print(f"{m:14d} | {mse:.4f} | {t_time:.2f}s | {i_time:.2f}s")